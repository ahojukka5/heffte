/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
       Value, gradient and Hessian of a periodic function: 3D transforms versus one direction at a time
*/

// Computes f, f_x, f_y, f_z, f_xx, f_yy, f_zz, f_xy, f_yz, f_zx on an n^3 grid over [0, 2 pi)^3
// for f = exp(g), g = sin(x) + cos(2y) / 2 + 0.4 sin(3z) + 0.3 sin(x + y) + 0.3 cos(y - z) + 0.2 sin(2x + z)
// and compares two methods, both with input and output in x-pencils:
//
//  3D:      one 3D r2c forward transform, then one 3D backward transform per field,
//           either one at a time or as a single batch of ten
//  pencils: a 1D derivative along the contiguous direction of each pencil layout,
//           x-pencils:                       f   -> f_x, f_xx
//           reshape f, f_x to y-pencils:     f   -> f_y, f_yy;  f_x -> f_xy
//           reshape f, f_x, f_y to z-pencils: f  -> f_z, f_zz;  f_x -> f_zx;  f_y -> f_yz
//           then optionally reshape the y and z results back to x-pencils,
//           each reshape moves all its fields with one batched call
//
// usage: mpirun -np <ranks> derivatives3d [n] [runs]

#include "heffte.h"

using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
using executor_r2c = heffte::one_dim_backend<backend_tag>::executor_r2c;

//! \brief Holds the exponent g and its first and second derivatives at one point.
struct exponent{
    double g, gx, gy, gz, gxx, gyy, gzz, gxy, gyz, gzx;
};

exponent make_exponent(double x, double y, double z){
    return {
        std::sin(x) + 0.5 * std::cos(2.0 * y) + 0.4 * std::sin(3.0 * z)
            + 0.3 * std::sin(x + y) + 0.3 * std::cos(y - z) + 0.2 * std::sin(2.0 * x + z),
        std::cos(x) + 0.3 * std::cos(x + y) + 0.4 * std::cos(2.0 * x + z),
        -std::sin(2.0 * y) + 0.3 * std::cos(x + y) - 0.3 * std::sin(y - z),
        1.2 * std::cos(3.0 * z) + 0.3 * std::sin(y - z) + 0.2 * std::cos(2.0 * x + z),
        -std::sin(x) - 0.3 * std::sin(x + y) - 0.8 * std::sin(2.0 * x + z),
        -2.0 * std::cos(2.0 * y) - 0.3 * std::sin(x + y) - 0.3 * std::cos(y - z),
        -3.6 * std::sin(3.0 * z) - 0.3 * std::cos(y - z) - 0.2 * std::sin(2.0 * x + z),
        -0.3 * std::sin(x + y),
        0.3 * std::cos(y - z),
        -0.4 * std::sin(2.0 * x + z)
    };
}

//! \brief Factor (i k)^power / scale, the Nyquist mode is dropped from the odd powers.
std::complex<double> spectral_factor(int k, int n, int power, double scale){
    if (power % 2 == 1 and 2 * std::abs(k) == n) return 0.0;
    std::complex<double> r = 1.0 / scale;
    for(int p = 0; p < power; p++) r *= std::complex<double>(0.0, k);
    return r;
}

/*!
 * \brief Derivatives along the contiguous direction of a pencil, one r2c and one c2r per requested power.
 */
struct pencil_stage{
    pencil_stage(heffte::box3d<> const box, int dimension) :
        n(box.size[dimension]), line(n / 2 + 1),
        executor(heffte::make_executor_r2c<backend_tag>(nullptr, box, dimension)),
        fhat(executor->complex_size()), scratch(executor->complex_size()), workspace(executor->workspace_size())
    {
        if (box.order[0] != dimension) throw std::runtime_error("the pencil direction must be contiguous");
        for(int power = 0; power <= 2; power++)
            for(int c = 0; c < line; c++)
                factors[power].push_back(spectral_factor(c, n, power, n));
    }
    //! \brief Computes the derivatives of the given powers of the field and writes them into the outputs.
    void apply(double const field[], std::vector<std::pair<int, double*>> const &outputs){
        executor->forward(field, fhat.data(), workspace.data());
        for(auto const &out : outputs){
            std::vector<std::complex<double>> const &factor = factors[out.first];
            for(size_t i = 0; i < fhat.size(); i++)
                scratch[i] = factor[i % line] * fhat[i];
            executor->backward(scratch.data(), out.second, workspace.data());
        }
    }
    int n, line;
    std::unique_ptr<executor_r2c> executor;
    std::vector<std::complex<double>> fhat, scratch, workspace;
    std::array<std::vector<std::complex<double>>, 3> factors;
};

//! \brief Applies a reshape that moves the given number of fields, a null reshape is a copy.
void reshape(std::unique_ptr<heffte::reshape3d_base<int>> const &shaper, int batch_size, double const source[], double destination[],
             int source_size, std::vector<double> &workspace){
    if (shaper){
        shaper->apply(batch_size, source, destination, workspace.data());
    }else{
        std::copy(source, source + batch_size * source_size, destination);
    }
}

void benchmark_derivatives(int n, int num_runs, MPI_Comm comm){

    int const me        = heffte::mpi::comm_rank(comm);
    int const num_ranks = heffte::mpi::comm_size(comm);

    double const h = 2.0 * 3.14159265358979323846 / n;

    heffte::box3d<> const world({0, 0, 0}, {n - 1, n - 1, n - 1});
    std::array<int, 2> const grid = heffte::make_procgrid(num_ranks);

    // pencil layouts, each with its pencil direction contiguous in memory
    std::vector<heffte::box3d<>> const xpencils = heffte::split_world(world, {1, grid[0], grid[1]});
    std::vector<heffte::box3d<>> const ypencils = heffte::make_pencils(world, grid, 1, xpencils, {1, 0, 2});
    std::vector<heffte::box3d<>> const zpencils = heffte::make_pencils(world, grid, 2, ypencils, {2, 0, 1});
    heffte::box3d<> const xbox = xpencils[me], ybox = ypencils[me], zbox = zpencils[me];
    int const nx = xbox.count(), ny = ybox.count(), nz = zbox.count();

    // the field and the exact derivatives in the x-pencils, ordered as the output of both methods
    std::array<std::string, 10> const names = {"f", "f_x", "f_y", "f_z", "f_xx", "f_yy", "f_zz", "f_xy", "f_yz", "f_zx"};
    std::array<std::array<int, 3>, 10> const orders = {{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
        {1, 1, 0}, {0, 1, 1}, {1, 0, 1}
    }};
    std::vector<double> input(nx);
    std::array<std::vector<double>, 10> exact;
    for(auto &e : exact) e.resize(nx);
    int idx = 0;
    for(int k = xbox.low[2]; k <= xbox.high[2]; k++){
        for(int j = xbox.low[1]; j <= xbox.high[1]; j++){
            for(int i = xbox.low[0]; i <= xbox.high[0]; i++){
                exponent const e = make_exponent(i * h, j * h, k * h);
                double const f = std::exp(e.g);
                input[idx] = f;
                std::array<double, 10> const values = {
                    f, f * e.gx, f * e.gy, f * e.gz,
                    f * (e.gxx + e.gx * e.gx), f * (e.gyy + e.gy * e.gy), f * (e.gzz + e.gz * e.gz),
                    f * (e.gxy + e.gx * e.gy), f * (e.gyz + e.gy * e.gz), f * (e.gzx + e.gz * e.gx)};
                for(int d = 0; d < 10; d++) exact[d][idx] = values[d];
                idx++;
            }
        }
    }

    // the error of each field relative to its largest exact value, reduced over all ranks
    auto max_error = [&](std::array<double const*, 10> const &computed)->std::array<double, 10>{
        std::array<double, 10> result;
        for(int d = 0; d < 10; d++){
            double local[2] = {0.0, 0.0};
            for(int i = 0; i < nx; i++){
                local[0] = std::max(local[0], std::abs(exact[d][i]));
                local[1] = std::max(local[1], std::abs(computed[d][i] - exact[d][i]));
            }
            double global[2];
            MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, comm);
            result[d] = global[1] / global[0];
        }
        return result;
    };

    // ----------------------------------------------------------------------------------------- 3D transforms
    heffte::box3d<> const cbox = heffte::split_world(world.r2c(0), {1, grid[0], grid[1]})[me];
    int const nc = cbox.count();

    std::vector<std::complex<double>> fhat(nc), scaled(10 * nc);
    std::vector<double> out3d(10 * nx);

    // the multipliers of the ten fields, including the 1 / n^3 scaling of the backward transform
    std::vector<std::complex<double>> multipliers(10 * nc);
    for(int d = 0; d < 10; d++){
        int c = 0;
        for(int k = cbox.low[2]; k <= cbox.high[2]; k++){
            int const kz = (2 * k <= n) ? k : k - n;
            for(int j = cbox.low[1]; j <= cbox.high[1]; j++){
                int const ky = (2 * j <= n) ? j : j - n;
                for(int i = cbox.low[0]; i <= cbox.high[0]; i++){
                    multipliers[d * nc + c] = spectral_factor(i, n, orders[d][0], double(n) * n * n)
                                            * spectral_factor(ky, n, orders[d][1], 1.0)
                                            * spectral_factor(kz, n, orders[d][2], 1.0);
                    c++;
                }
            }
        }
    }

    auto run_3d = [&](heffte::fft3d_r2c<backend_tag> const &fft, std::vector<std::complex<double>> &fft_workspace, bool batched){
        fft.forward(input.data(), fhat.data(), fft_workspace.data());
        for(int d = 0; d < 10; d++){
            for(int c = 0; c < nc; c++) scaled[d * nc + c] = multipliers[d * nc + c] * fhat[c];
            if (not batched) fft.backward(scaled.data() + d * nc, out3d.data() + d * nx, fft_workspace.data());
        }
        if (batched) fft.backward(10, scaled.data(), out3d.data(), fft_workspace.data());
    };

    // --------------------------------------------------------------------------------- one direction at a time
    pencil_stage xstage(xbox, 0), ystage(ybox, 1), zstage(zbox, 2);

    heffte::plan_options const options = heffte::default_options<backend_tag>();
    auto xy = heffte::make_reshape3d<backend_tag>(nullptr, xpencils, ypencils, comm, options);
    auto yz = heffte::make_reshape3d<backend_tag>(nullptr, ypencils, zpencils, comm, options);
    auto yx = heffte::make_reshape3d<backend_tag>(nullptr, ypencils, xpencils, comm, options);
    auto zx = heffte::make_reshape3d<backend_tag>(nullptr, zpencils, xpencils, comm, options);
    size_t workspace_size = 0;
    if (xy) workspace_size = std::max(workspace_size, 2 * xy->size_workspace());
    if (yz) workspace_size = std::max(workspace_size, 3 * yz->size_workspace());
    if (yx) workspace_size = std::max(workspace_size, 3 * yx->size_workspace());
    if (zx) workspace_size = std::max(workspace_size, 4 * zx->size_workspace());
    std::vector<double> reshape_workspace(workspace_size);

    // consecutive fields in each layout, so that every reshape moves a contiguous batch
    std::vector<double> xbuf(10 * nx); // f, f_x, f_xx, then f_y, f_yy, f_xy, f_z, f_zz, f_zx, f_yz returned
    std::vector<double> ybuf(5 * ny);  // f, f_x, f_y, f_yy, f_xy
    std::vector<double> zbuf(7 * nz);  // f, f_x, f_y, f_z, f_zz, f_zx, f_yz
    double time_fft = 0.0, time_reshape = 0.0;

    auto run_pencils = [&](bool return_to_x){
        double t0 = MPI_Wtime();
        std::copy(input.begin(), input.end(), xbuf.begin());
        xstage.apply(&xbuf[0], {{1, &xbuf[nx]}, {2, &xbuf[2 * nx]}});
        double t1 = MPI_Wtime();
        reshape(xy, 2, &xbuf[0], &ybuf[0], nx, reshape_workspace);
        double t2 = MPI_Wtime();
        ystage.apply(&ybuf[0], {{1, &ybuf[2 * ny]}, {2, &ybuf[3 * ny]}});
        ystage.apply(&ybuf[ny], {{1, &ybuf[4 * ny]}});
        double t3 = MPI_Wtime();
        reshape(yz, 3, &ybuf[0], &zbuf[0], ny, reshape_workspace);
        double t4 = MPI_Wtime();
        zstage.apply(&zbuf[0], {{1, &zbuf[3 * nz]}, {2, &zbuf[4 * nz]}});
        zstage.apply(&zbuf[nz], {{1, &zbuf[5 * nz]}});
        zstage.apply(&zbuf[2 * nz], {{1, &zbuf[6 * nz]}});
        double t5 = MPI_Wtime();
        if (return_to_x){
            reshape(yx, 3, &ybuf[2 * ny], &xbuf[3 * nx], ny, reshape_workspace);
            reshape(zx, 4, &zbuf[3 * nz], &xbuf[6 * nx], nz, reshape_workspace);
        }
        double t6 = MPI_Wtime();
        time_fft     += (t1 - t0) + (t3 - t2) + (t5 - t4);
        time_reshape += (t2 - t1) + (t4 - t3) + (t6 - t5);
    };

    // ----------------------------------------------------------------------------------------------- timing
    // average of one evaluation of all ten fields, slowest rank, after one warm-up run
    auto time_it = [&](auto &&run)->double{
        run();
        double total = 0.0;
        for(int r = 0; r < num_runs; r++){
            MPI_Barrier(comm);
            double t0 = MPI_Wtime();
            run();
            total += MPI_Wtime() - t0;
        }
        double local = total / num_runs, global;
        MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Bcast(&global, 1, MPI_DOUBLE, 0, comm);
        return global;
    };
    auto output_errors = [&](std::vector<double> const &out, std::array<int, 10> const &slot)->std::array<double, 10>{
        std::array<double const*, 10> fields;
        for(int d = 0; d < 10; d++) fields[d] = out.data() + slot[d] * nx;
        return max_error(fields);
    };
    double const tolerance = 1.E-10;
    auto passes = [&](std::array<double, 10> const &err)->bool{
        for(double e : err) if (not (e < tolerance)) return false;
        return true;
    };
    bool pass = true;

    if (me == 0) std::cout << "backend " << heffte::backend::name<backend_tag>() << ", grid " << n << "^3, "
                           << num_ranks << " rank(s), pencil grid " << grid[0] << " x " << grid[1] << "\n\n"
                           << "milliseconds per evaluation of all ten fields, average of " << num_runs << " runs\n\n"
                           << "  3D transforms, by plan options     one at a time   batched backward\n";

    // every combination of the plan options, the best one is the baseline
    std::array<int, 10> const in_order = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double best_3d = 1.E300;
    std::string best_name;
    for(bool use_pencils : {true, false}){
        for(bool use_reorder : {true, false}){
            for(heffte::reshape_algorithm algorithm : {heffte::reshape_algorithm::alltoallv, heffte::reshape_algorithm::alltoall,
                                                       heffte::reshape_algorithm::p2p_plined}){
                heffte::plan_options options = heffte::default_options<backend_tag>();
                options.use_pencils = use_pencils;
                options.use_reorder = use_reorder;
                options.algorithm   = algorithm;
                heffte::fft3d_r2c<backend_tag> fft(xbox, cbox, 0, comm, options);
                std::vector<std::complex<double>> fft_workspace(10 * fft.size_workspace());

                run_3d(fft, fft_workspace, false);
                bool const ok_single = passes(output_errors(out3d, in_order));
                run_3d(fft, fft_workspace, true);
                bool const ok_batch = passes(output_errors(out3d, in_order));
                pass = pass and ok_single and ok_batch;

                double const t_single = time_it([&]{ run_3d(fft, fft_workspace, false); });
                double const t_batch  = time_it([&]{ run_3d(fft, fft_workspace, true); });

                std::ostringstream name;
                name << (use_pencils ? "pencils" : "slabs  ") << (use_reorder ? ", reorder" : ", no reorder")
                     << ", " << ((algorithm == heffte::reshape_algorithm::alltoallv) ? "alltoallv" :
                                 ((algorithm == heffte::reshape_algorithm::alltoall) ? "alltoall" : "p2p pipelined"));
                for(auto t : {std::make_pair(t_single, std::string(", one at a time")), std::make_pair(t_batch, std::string(", batched"))}){
                    if (t.first < best_3d){
                        best_3d   = t.first;
                        best_name = name.str() + t.second;
                    }
                }
                if (me == 0) std::cout << "  " << std::left << std::setw(36) << name.str() << std::right << std::fixed << std::setprecision(2)
                                       << std::setw(10) << 1.E3 * t_single << std::setw(17) << 1.E3 * t_batch
                                       << ((ok_single and ok_batch) ? "" : "   FAILED") << "\n";
            }
        }
    }

    // pencils, verified after the return to x-pencils,
    // xbuf holds f, f_x, f_xx, f_y, f_yy, f_xy, f_z, f_zz, f_zx, f_yz
    run_pencils(true);
    std::array<double, 10> const err_pencils = output_errors(xbuf, {0, 1, 3, 6, 2, 4, 7, 5, 9, 8});
    pass = pass and passes(err_pencils);

    double const t_inplace  = time_it([&]{ run_pencils(false); });
    time_fft = time_reshape = 0.0;
    double const t_returned = time_it([&]{ run_pencils(true); });
    double split[2] = {time_fft / (num_runs + 1), time_reshape / (num_runs + 1)}, split_max[2];
    MPI_Reduce(split, split_max, 2, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (me == 0){
        std::cout << std::fixed << std::setprecision(2)
                  << "\n  best 3D: " << best_name << ", " << 1.E3 * best_3d << "\n\n"
                  << "  one direction at a time\n"
                  << "    results left in the pencils:    " << std::setw(8) << 1.E3 * t_inplace
                  << "   (" << best_3d / t_inplace << "x faster than the best 3D)\n"
                  << "    results back in x-pencils:      " << std::setw(8) << 1.E3 * t_returned
                  << "   (" << best_3d / t_returned << "x faster than the best 3D)\n"
                  << "      of which 1D transforms:       " << std::setw(8) << 1.E3 * split_max[0] << "\n"
                  << "      of which reshapes:            " << std::setw(8) << 1.E3 * split_max[1] << "\n\n"
                  << std::scientific << std::setprecision(2)
                  << "  relative errors, one direction at a time\n";
        for(int d = 0; d < 10; d++) std::cout << "    " << std::setw(5) << names[d] << "  " << err_pencils[d] << "\n";
        std::cout << "\n" << (pass ? "all relative errors below " : "FAILED, an error exceeds ") << tolerance << "\n";
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int const n        = (argc > 1) ? std::stoi(argv[1]) : 128;
    int const num_runs = (argc > 2) ? std::stoi(argv[2]) : 10;

    benchmark_derivatives(n, num_runs, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
