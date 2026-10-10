#include "heffte.h"

/*!
 * \brief HeFFTe example 7, spectral derivatives of a periodic function: the value, the gradient and the Hessian.
 *
 * The function is sampled on a 128 by 128 by 128 grid over the periodic box [0, 2 pi)^3
 * split across an arbitrary number of MPI ranks.
 * The test function is a three dimensional relative of exp(sin(x)) from Trefethen's
 * Spectral Methods in MATLAB, f = exp(g) where
 *     g = sin(x) + cos(2y) / 2 + 0.4 sin(3z) + 0.3 sin(x + y) + 0.3 cos(y - z) + 0.2 sin(2x + z)
 * Each direction has a different frequency and the mixed terms couple every pair of directions,
 * so a derivative taken in the wrong direction or with the wrong sign shows up in the error.
 * The exact derivatives are f_i = f g_i and f_ij = f (g_ij + g_i g_j).
 *
 * One forward real-to-complex transform gives the coefficients of f, each derivative
 * multiplies the coefficients by (i k_x)^a (i k_y)^b (i k_z)^c and takes one backward transform.
 * The Nyquist mode is dropped from odd derivatives, since its derivative is not real.
 */

//! \brief Holds the function g and its first and second derivatives at one point.
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

void compute_derivatives(MPI_Comm comm){

    using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;

    int const me        = heffte::mpi::comm_rank(comm);
    int const num_ranks = heffte::mpi::comm_size(comm);

    if (me == 0) std::cout << "using backend: " << heffte::backend::name<backend_tag>() << "\n";

    int const n = 128;
    double const pi = 3.14159265358979323846;
    double const h  = 2.0 * pi / n;

    // the r2c transform shrinks direction 0, the complex indexes are (n / 2 + 1) x n x n
    int const r2c_direction = 0;
    heffte::box3d<> const real_indexes({0, 0, 0}, {n - 1, n - 1, n - 1});
    heffte::box3d<> const complex_indexes = real_indexes.r2c(r2c_direction);

    std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(real_indexes, num_ranks);

    heffte::box3d<> const inbox  = heffte::split_world(real_indexes,    proc_grid)[me];
    heffte::box3d<> const outbox = heffte::split_world(complex_indexes, proc_grid)[me];

    heffte::fft3d_r2c<backend_tag> fft(inbox, outbox, r2c_direction, comm);

    // the derivatives as the powers of (i k_x, i k_y, i k_z), the first entry is the value itself
    std::array<std::string, 10> const names = {"f", "f_x", "f_y", "f_z", "f_xx", "f_yy", "f_zz", "f_xy", "f_yz", "f_zx"};
    std::array<std::array<int, 3>, 10> const orders = {{
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
        {1, 1, 0}, {0, 1, 1}, {1, 0, 1}
    }};

    // sample f and the exact derivatives, the data is ordered with direction 0 the fastest
    std::vector<double> input(fft.size_inbox());
    std::array<std::vector<double>, 10> exact;
    for(auto &e : exact) e.resize(fft.size_inbox());

    int idx = 0;
    for(int k = inbox.low[2]; k <= inbox.high[2]; k++){
        for(int j = inbox.low[1]; j <= inbox.high[1]; j++){
            for(int i = inbox.low[0]; i <= inbox.high[0]; i++){
                exponent const e = make_exponent(i * h, j * h, k * h);
                double const f = std::exp(e.g);
                input[idx] = f;
                exact[0][idx] = f;
                exact[1][idx] = f * e.gx;
                exact[2][idx] = f * e.gy;
                exact[3][idx] = f * e.gz;
                exact[4][idx] = f * (e.gxx + e.gx * e.gx);
                exact[5][idx] = f * (e.gyy + e.gy * e.gy);
                exact[6][idx] = f * (e.gzz + e.gz * e.gz);
                exact[7][idx] = f * (e.gxy + e.gx * e.gy);
                exact[8][idx] = f * (e.gyz + e.gy * e.gz);
                exact[9][idx] = f * (e.gzx + e.gz * e.gx);
                idx++;
            }
        }
    }

    // the factor k^power of a global index, the Nyquist index n / 2 is kept for the even powers only
    auto wavenumber = [&](int index, int power)->double{
        if (power % 2 == 1 and 2 * index == n) return 0.0;
        double const k = (2 * index <= n) ? index : index - n;
        return (power == 0) ? 1.0 : ((power == 1) ? k : k * k);
    };

    // multiplies the coefficients by (i k_x)^a (i k_y)^b (i k_z)^c for one derivative
    auto differentiate = [&](std::vector<std::complex<double>> const &fhat, std::array<int, 3> const &order,
                             std::vector<std::complex<double>> &result){
        int const power = order[0] + order[1] + order[2];
        // i^power is 1, i, -1, -i
        std::complex<double> const ipow = std::array<std::complex<double>, 4>{{{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}}[power % 4];
        int c = 0;
        for(int k = outbox.low[2]; k <= outbox.high[2]; k++){
            double const kz = wavenumber(k, order[2]);
            for(int j = outbox.low[1]; j <= outbox.high[1]; j++){
                double const kyz = kz * wavenumber(j, order[1]);
                for(int i = outbox.low[0]; i <= outbox.high[0]; i++){
                    result[c] = ipow * (kyz * wavenumber(i, order[0])) * fhat[c];
                    c++;
                }
            }
        }
    };

    std::vector<std::complex<double>> fhat(fft.size_outbox());
    std::vector<std::complex<double>> scratch(fft.size_outbox());
    std::array<std::vector<double>, 10> computed;
    for(auto &d : computed) d.resize(fft.size_inbox());

    // one evaluation of everything: one forward transform, ten multiplications and ten backward transforms
    // the timers are accumulated over the runs, the first run is a warm up and not counted
    double time_forward = 0.0, time_multiply = 0.0, time_backward = 0.0;
    int const num_runs = 10;
    for(int run = 0; run <= num_runs; run++){
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        fft.forward(input.data(), fhat.data());
        double t1 = MPI_Wtime();
        double multiply = 0.0, backward = 0.0;
        for(size_t d = 0; d < orders.size(); d++){
            double s0 = MPI_Wtime();
            differentiate(fhat, orders[d], scratch);
            double s1 = MPI_Wtime();
            fft.backward(scratch.data(), computed[d].data(), heffte::scale::full);
            double s2 = MPI_Wtime();
            multiply += s1 - s0;
            backward += s2 - s1;
        }
        if (run > 0){
            time_forward  += t1 - t0;
            time_multiply += multiply;
            time_backward += backward;
        }
    }

    // the error in each derivative relative to the largest exact value, reduced over all ranks
    bool pass = true;
    double const tolerance = 1.E-10;
    if (me == 0) std::cout << std::scientific << std::setprecision(3) << "\n"
                           << "grid " << n << "^3 on " << num_ranks << " rank(s)\n\n"
                           << " derivative   max |exact|   max |error|   relative error\n";
    for(size_t d = 0; d < orders.size(); d++){
        double local[2] = {0.0, 0.0};
        for(size_t i = 0; i < input.size(); i++){
            local[0] = std::max(local[0], std::abs(exact[d][i]));
            local[1] = std::max(local[1], std::abs(computed[d][i] - exact[d][i]));
        }
        double global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, comm);
        double const relative = global[1] / global[0];
        pass = pass and relative < tolerance;
        if (me == 0) std::cout << std::setw(11) << names[d] << std::setw(14) << global[0]
                               << std::setw(14) << global[1] << std::setw(17) << relative << "\n";
    }

    // report the slowest rank, the time is the average of one evaluation of all ten fields
    double local_times[3] = {time_forward / num_runs, time_multiply / num_runs, time_backward / num_runs};
    double times[3];
    MPI_Reduce(local_times, times, 3, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (me == 0){
        std::cout << "\naverage over " << num_runs << " runs, seconds\n"
                  << "  forward transform (1):    " << times[0] << "\n"
                  << "  spectral multiply (10):   " << times[1] << "\n"
                  << "  backward transforms (10): " << times[2] << "\n"
                  << "  total:                    " << times[0] + times[1] + times[2] << "\n\n"
                  << (pass ? "all relative errors below " : "FAILED, an error exceeds ") << tolerance << "\n";
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_derivatives(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
