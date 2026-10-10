/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
       Spectral derivatives of a periodic function: 3D transforms versus one direction at a time
*/

// Computes derivatives such as f_x, f_yy or f_xy on an n^3 grid over [0, 2 pi)^3 for
// f = exp(g), g = sin(x) + cos(2y) / 2 + 0.4 sin(3z) + 0.3 sin(x + y) + 0.3 cos(y - z) + 0.2 sin(2x + z),
// for single components and for sets (gradient, gradient + Hessian diagonal, gradient + Hessian),
// and times four methods; every result returns to the input layout:
//
//  3D pencils:  fft3d_r2c from x-pencils with the spectrum left in z-pencils, one forward and
//               one backward transform per field (one at a time or batched)
//  1D pencils:  1D transforms along the local direction of x, y and z pencils
//  3D slabs:    fft3d_r2c from xy-slabs with the spectrum left in a z-local layout
//  1D slabs:    1D transforms along the local directions of xy, yz and xz slabs
//
// The 1D methods take the derivative along the local directions of each layout they visit,
// share intermediates (f_x is also the input of f_xy), move each field once to each layout
// that needs it, and move all fields between the same two layouts with one batched reshape.
// The route through the layouts is the one that moves the fewest fields, then makes the fewest
// reshape calls, then takes the fewest 1D transforms.
//
// The plan options of the 3D methods (reorder, reshape algorithm, batched backward) and the
// reshape algorithm of the 1D methods are chosen by timing the gradient + Hessian set,
// then kept for every set.
//
// With DERIVATIVES3D_ROCM defined it runs on rocFFT with device buffers and GPU-aware MPI,
// linked with derivatives3d_kernels.hip, one GPU per rank.
//
// usage: mpirun -np <ranks> derivatives3d [n] [runs]
//        prints a table and one CSV line per set and method (lines starting with CSV)

#include <cstring>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include "heffte.h"

#ifdef DERIVATIVES3D_ROCM
// GPU build: rocFFT, device buffers, GPU-aware reshapes; the kernels are in derivatives3d_kernels.hip
using backend_tag = heffte::backend::rocfft;
template<typename T> using device_vector = heffte::gpu::vector<T>;
void device_multiply(long n, std::complex<double> const factor[], std::complex<double> const in[], std::complex<double> out[]);
void device_copy(void *destination, void const *source, size_t bytes);
void device_sync();
#else
using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
template<typename T> using device_vector = std::vector<T>;
inline void device_multiply(long n, std::complex<double> const factor[], std::complex<double> const in[], std::complex<double> out[]){
    for(long i = 0; i < n; i++) out[i] = factor[i] * in[i];
}
inline void device_copy(void *destination, void const *source, size_t bytes){ std::memcpy(destination, source, bytes); }
inline void device_sync(){}
#endif

template<typename T> device_vector<T> to_device(std::vector<T> const &host){
    device_vector<T> result(host.size());
    device_copy(result.data(), host.data(), host.size() * sizeof(T));
    return result;
}
template<typename T> std::vector<T> to_host(T const device[], size_t size){
    std::vector<T> result(size);
    device_copy(result.data(), device, size * sizeof(T));
    return result;
}

using executor_r2c = heffte::one_dim_backend<backend_tag>::executor_r2c;
using reshaper     = std::unique_ptr<heffte::reshape3d_base<int>>;
using order3       = std::array<int, 3>;

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

//! \brief The exact value of f or of one of its first or second derivatives, false when the order is higher.
bool exact_value(order3 const &d, double x, double y, double z, double &value){
    exponent const e = make_exponent(x, y, z);
    double const f = std::exp(e.g);
    int const total = d[0] + d[1] + d[2];
    if (total == 0){ value = f; return true; }
    std::array<double, 3> const g1 = {e.gx, e.gy, e.gz};
    if (total == 1){
        for(int a = 0; a < 3; a++) if (d[a] == 1) value = f * g1[a];
        return true;
    }
    if (total == 2){
        std::array<std::array<double, 3>, 3> const g2 = {{{e.gxx, e.gxy, e.gzx}, {e.gxy, e.gyy, e.gyz}, {e.gzx, e.gyz, e.gzz}}};
        int a = -1, b = -1;
        for(int i = 0; i < 3; i++) for(int k = 0; k < d[i]; k++) ((a < 0) ? a : b) = i;
        value = f * (g2[a][b] + g1[a] * g1[b]);
        return true;
    }
    return false;
}

//! \brief Factor (i k)^power / scale, the Nyquist mode is dropped from the odd powers.
std::complex<double> spectral_factor(int k, int n, int power, double scale){
    if (power % 2 == 1 and 2 * std::abs(k) == n) return 0.0;
    std::complex<double> r = 1.0 / scale;
    for(int p = 0; p < power; p++) r *= std::complex<double>(0.0, k);
    return r;
}

//! \brief Calls the function with the global (i, j, k) of every entry of the box, in memory order.
template<typename callback>
void for_each_index(heffte::box3d<> const &box, callback &&fn){
    std::array<int, 3> idx;
    for(int s = 0; s < box.osize(2); s++){
        idx[box.order[2]] = box.low[box.order[2]] + s;
        for(int m = 0; m < box.osize(1); m++){
            idx[box.order[1]] = box.low[box.order[1]] + m;
            for(int q = 0; q < box.osize(0); q++){
                idx[box.order[0]] = box.low[box.order[0]] + q;
                fn(idx);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------- layouts
enum layout_id{ X = 0, Y, Z, XY, YZ, XZ, num_layouts };
std::array<std::string, num_layouts> const layout_names = {"X", "Y", "Z", "XY", "YZ", "XZ"};
std::array<std::vector<int>, num_layouts> const local_axes = {{{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}}};

//! \brief The boxes of every layout, each pairing ranks to reduce the data moved from the previous one.
std::array<std::vector<heffte::box3d<>>, num_layouts> make_layouts(heffte::box3d<> const world, int num_ranks){
    std::array<int, 2> const grid = heffte::make_procgrid(num_ranks);
    std::array<std::vector<heffte::box3d<>>, num_layouts> boxes;
    boxes[X]  = heffte::split_world(world, {1, grid[0], grid[1]});
    boxes[Y]  = heffte::make_pencils(world, grid, 1, boxes[X], {1, 0, 2});
    boxes[Z]  = heffte::make_pencils(world, grid, 2, boxes[Y], {2, 0, 1});
    boxes[XY] = heffte::make_slabs(world, num_ranks, 0, 1, boxes[X], {0, 1, 2}, heffte::rank_remap());
    boxes[YZ] = heffte::make_slabs(world, num_ranks, 1, 2, boxes[XY], {1, 2, 0}, heffte::rank_remap());
    boxes[XZ] = heffte::make_slabs(world, num_ranks, 0, 2, boxes[XY], {0, 2, 1}, heffte::rank_remap());
    return boxes;
}

// --------------------------------------------------------------------------------- the 1D plan
using field_key = order3;   // the derivative orders applied so far

/*!
 * \brief A plan for the one-direction-at-a-time method.
 *
 * Each stage moves in the fields it needs, then transforms along its local directions in increasing
 * order, so an intermediate made along one direction can feed the next direction in the same stage.
 */
struct plan_1d{
    struct derivative{ int power; field_key output; };
    struct forward{ field_key input; int axis; std::vector<derivative> outputs; };
    struct stage{ int layout; std::vector<std::pair<int, std::vector<field_key>>> moves_in; std::vector<forward> transforms; };
    int home = X;
    std::vector<int> route;
    std::vector<stage> stages;
    std::vector<std::pair<int, std::vector<field_key>>> moves_home;
    int passes = 0, moved = 0, calls = 0;
    bool valid = false;
};

plan_1d make_plan(std::vector<order3> const &components, int home, std::vector<int> const &route){
    plan_1d p;
    p.home = home;
    p.route = route;
    std::map<field_key, std::set<int>> present;
    std::map<field_key, int> origin;
    present[{0, 0, 0}] = {home};
    origin[{0, 0, 0}] = home;
    std::map<order3, field_key> state;
    for(auto const &c : components) state[c] = {0, 0, 0};

    std::vector<int> layouts = {home};
    layouts.insert(layouts.end(), route.begin(), route.end());
    for(int layout : layouts){
        plan_1d::stage st;
        st.layout = layout;
        std::map<int, std::vector<field_key>> moves_in;
        for(int axis : local_axes[layout]){
            std::map<field_key, plan_1d::forward> forwards;
            for(auto const &c : components){
                field_key k = state[c];
                if (c[axis] == k[axis]) continue;
                if (not present[k].count(layout)){
                    moves_in[origin[k]].push_back(k);
                    present[k].insert(layout);
                }
                field_key out = k;
                out[axis] = c[axis];
                auto &fw = forwards[k];
                fw.input = k;
                fw.axis = axis;
                bool known = false;
                for(auto const &o : fw.outputs) if (o.output == out) known = true;
                if (not known){
                    if (present.count(out)) return p; // computed elsewhere, this route is not useful
                    fw.outputs.push_back({c[axis], out});
                    present[out] = {layout};
                    origin[out] = layout;
                }
                state[c] = out;
            }
            for(auto &fw : forwards){
                st.transforms.push_back(fw.second);
                p.passes += 1 + static_cast<int>(fw.second.outputs.size());
            }
        }
        for(auto &m : moves_in){
            st.moves_in.push_back(m);
            p.moved += static_cast<int>(m.second.size());
            p.calls++;
        }
        p.stages.push_back(st);
    }
    for(auto const &c : components) if (state[c] != c) return p;
    std::map<int, std::vector<field_key>> moves_home;
    for(auto const &c : components){
        if (not present[c].count(home)){
            moves_home[origin[c]].push_back(c);
            present[c].insert(home);
        }
    }
    for(auto &m : moves_home){
        p.moves_home.push_back(m);
        p.moved += static_cast<int>(m.second.size());
        p.calls++;
    }
    p.valid = true;
    return p;
}

//! \brief Fewest moved fields, then fewest reshape calls, then fewest 1D transforms, over every route.
plan_1d best_plan(std::vector<order3> const &components, int home, std::vector<int> const &family){
    std::vector<int> others;
    for(int l : family) if (l != home) others.push_back(l);
    plan_1d best;
    auto better = [](plan_1d const &a, plan_1d const &b){
        return std::make_tuple(a.moved, a.calls, a.passes, a.route.size())
             < std::make_tuple(b.moved, b.calls, b.passes, b.route.size());
    };
    int const m = static_cast<int>(others.size());
    for(int mask = 0; mask < (1 << m); mask++){
        std::vector<int> subset;
        for(int i = 0; i < m; i++) if (mask & (1 << i)) subset.push_back(others[i]);
        std::sort(subset.begin(), subset.end());
        do{
            plan_1d p = make_plan(components, home, subset);
            if (p.valid and (not best.valid or better(p, best))) best = p;
        }while(std::next_permutation(subset.begin(), subset.end()));
    }
    return best;
}

/*!
 * \brief Executes a 1D plan: owns the fields in every layout, the executors and the reshapes.
 */
struct runner_1d{
    runner_1d(plan_1d const &plan, std::array<std::vector<heffte::box3d<>>, num_layouts> const &boxes,
              heffte::reshape_algorithm algorithm, int n, MPI_Comm comm) : p(plan), n(n), me(heffte::mpi::comm_rank(comm)){
        heffte::plan_options options = heffte::default_options<backend_tag>();
        options.algorithm = algorithm;
        for(int l = 0; l < num_layouts; l++){
            box.push_back(boxes[l][me]);
            count[l] = box[l].count();
        }
        auto add_shaper = [&](int src, int dst, int batch){
            auto key = std::make_pair(src, dst);
            if (not shapers.count(key))
                shapers[key] = heffte::make_reshape3d<backend_tag>(nullptr, boxes[src], boxes[dst], comm, options);
            if (shapers[key]) workspace_size = std::max(workspace_size, batch * shapers[key]->size_workspace());
            staging_size = std::max(staging_size, static_cast<size_t>(batch) * std::max(count[src], count[dst]));
        };
        // the received fields of a batch share one block, so the reshape writes them in place
        auto receive = [&](int dst, std::vector<field_key> const &keys){
            blocks.emplace_back(keys.size() * count[dst]);
            for(size_t i = 0; i < keys.size(); i++) storage[{dst, keys[i]}] = blocks.back().data() + i * count[dst];
        };
        auto produce = [&](int layout, field_key const &k){
            if (storage.count({layout, k})) return;
            blocks.emplace_back(count[layout]);
            storage[{layout, k}] = blocks.back().data();
        };
        // the fields computed in a layout that leave it together are allocated next to each other
        std::map<std::pair<int, field_key>, bool> received;
        for(auto const &st : p.stages) for(auto const &m : st.moves_in) for(auto const &k : m.second) received[{st.layout, k}] = true;
        auto group_outgoing = [&](int layout, std::vector<field_key> const &keys){
            for(auto const &k : keys) if (storage.count({layout, k}) or received.count({layout, k})) return;
            blocks.emplace_back(keys.size() * count[layout]);
            for(size_t i = 0; i < keys.size(); i++) storage[{layout, keys[i]}] = blocks.back().data() + i * count[layout];
        };
        for(auto const &m : p.moves_home) group_outgoing(m.first, m.second);
        for(auto const &st : p.stages) for(auto const &m : st.moves_in) group_outgoing(m.first, m.second);
        produce(p.home, {0, 0, 0});
        for(auto const &st : p.stages){
            for(auto const &m : st.moves_in){
                add_shaper(m.first, st.layout, static_cast<int>(m.second.size()));
                receive(st.layout, m.second);
            }
            for(auto const &fw : st.transforms){
                auto key = std::make_pair(st.layout, fw.axis);
                // a direction that is not contiguous is transposed locally first, a strided 1D
                // transform is several times slower than the transpose
                heffte::box3d<> const &lbox = box[st.layout];
                order3 order = {fw.axis, 0, 0};
                int j = 1;
                for(int a : lbox.order) if (a != fw.axis) order[j++] = a;
                heffte::box3d<> const tbox(lbox.low, lbox.high, order);
                if (lbox.order[0] != fw.axis){
                    auto const tboxes = heffte::reorder(boxes[st.layout], order);
                    if (not executors.count(key)){
                        to_contiguous[key]   = heffte::make_reshape3d<backend_tag>(nullptr, boxes[st.layout], tboxes, comm, options);
                        from_contiguous[key] = heffte::make_reshape3d<backend_tag>(nullptr, tboxes, boxes[st.layout], comm, options);
                        for(auto const *t : {&to_contiguous[key], &from_contiguous[key]})
                            if (*t) workspace_size = std::max(workspace_size, (*t)->size_workspace());
                        transposed_in  = device_vector<double>(count[st.layout]);
                        transposed_out = device_vector<double>(count[st.layout]);
                    }
                }
                if (not executors.count(key)){
                    executors[key] = heffte::make_executor_r2c<backend_tag>(nullptr, tbox, fw.axis);
                    fhat[key]    = device_vector<std::complex<double>>(executors[key]->complex_size());
                    scratch[key] = device_vector<std::complex<double>>(executors[key]->complex_size());
                    exec_workspace_size = std::max(exec_workspace_size, executors[key]->workspace_size());
                }
                heffte::box3d<> const cbox = tbox.r2c(fw.axis);
                for(auto const &o : fw.outputs){
                    auto fkey = std::make_tuple(st.layout, fw.axis, o.power);
                    if (not factors.count(fkey)){
                        std::vector<std::complex<double>> line(n / 2 + 1);
                        for(int c = 0; c <= n / 2; c++) line[c] = spectral_factor(c, n, o.power, n);
                        // the factor of every complex entry, in memory order
                        std::vector<std::complex<double>> fac;
                        fac.reserve(cbox.count());
                        for_each_index(cbox, [&](std::array<int, 3> const &idx){ fac.push_back(line[idx[fw.axis]]); });
                        factors[fkey] = to_device(fac);
                    }
                    produce(st.layout, o.output);
                }
            }
        }
        for(auto const &m : p.moves_home){
            add_shaper(m.first, p.home, static_cast<int>(m.second.size()));
            receive(p.home, m.second);
        }
        reshape_workspace = device_vector<double>(workspace_size);
        staging           = device_vector<double>(staging_size);
        exec_workspace    = device_vector<std::complex<double>>(exec_workspace_size);
    }
    double* field(int layout, field_key const &k){ return storage.at({layout, k}); }

    void transpose(reshaper const &t, double const source[], double destination[], int size){
        if (t) t->apply(1, source, destination, reshape_workspace.data());
        else device_copy(destination, source, size * sizeof(double));
    }

    void move(int src, int dst, std::vector<field_key> const &keys){
        int const batch = static_cast<int>(keys.size());
        double const *source = field(src, keys[0]);
        bool contiguous = true;
        for(int i = 1; i < batch; i++) contiguous = contiguous and field(src, keys[i]) == field(src, keys[0]) + i * count[src];
        if (not contiguous){
            for(int i = 0; i < batch; i++) device_copy(staging.data() + i * count[src], field(src, keys[i]), count[src] * sizeof(double));
            source = staging.data();
        }
        reshaper const &s = shapers.at({src, dst});
        if (s) s->apply(batch, source, field(dst, keys[0]), reshape_workspace.data());
        else device_copy(field(dst, keys[0]), source, batch * count[src] * sizeof(double));
    }
    //! \brief One evaluation, accumulates the time of the transforms and of the reshapes.
    void run(){
        for(auto const &st : p.stages){
            double t0 = MPI_Wtime();
            for(auto const &m : st.moves_in) move(m.first, st.layout, m.second);
            device_sync();
            double t1 = MPI_Wtime();
            for(auto const &fw : st.transforms){
                auto key = std::make_pair(st.layout, fw.axis);
                auto &exec = executors[key];
                auto &fh = fhat[key];
                auto &sc = scratch[key];
                bool const transposed = to_contiguous.count(key) > 0;
                if (transposed){
                    transpose(to_contiguous[key], field(st.layout, fw.input), transposed_in.data(), count[st.layout]);
                    exec->forward(transposed_in.data(), fh.data(), exec_workspace.data());
                }else{
                    exec->forward(field(st.layout, fw.input), fh.data(), exec_workspace.data());
                }
                for(auto const &o : fw.outputs){
                    auto const &fac = factors[std::make_tuple(st.layout, fw.axis, o.power)];
                    device_multiply(static_cast<long>(fh.size()), fac.data(), fh.data(), sc.data());
                    if (transposed){
                        exec->backward(sc.data(), transposed_out.data(), exec_workspace.data());
                        transpose(from_contiguous[key], transposed_out.data(), field(st.layout, o.output), count[st.layout]);
                    }else{
                        exec->backward(sc.data(), field(st.layout, o.output), exec_workspace.data());
                    }
                }
            }
            device_sync();
            double t2 = MPI_Wtime();
            time_reshape   += t1 - t0;
            time_transform += t2 - t1;
        }
        double t0 = MPI_Wtime();
        for(auto const &m : p.moves_home) move(m.first, p.home, m.second);
        device_sync();
        time_reshape += MPI_Wtime() - t0;
    }

    plan_1d p;
    int n, me;
    std::vector<heffte::box3d<>> box;
    std::array<int, num_layouts> count = {};
    std::map<std::pair<int, int>, reshaper> shapers, to_contiguous, from_contiguous;
    std::map<std::pair<int, int>, std::unique_ptr<executor_r2c>> executors;
    std::map<std::pair<int, int>, device_vector<std::complex<double>>> fhat, scratch;
    std::map<std::tuple<int, int, int>, device_vector<std::complex<double>>> factors;
    std::map<std::pair<int, field_key>, double*> storage;
    std::list<device_vector<double>> blocks;
    size_t workspace_size = 0, staging_size = 0, exec_workspace_size = 0;
    device_vector<double> reshape_workspace, staging, transposed_in, transposed_out;
    device_vector<std::complex<double>> exec_workspace;
    double time_transform = 0.0, time_reshape = 0.0;
};

// ----------------------------------------------------------------------------- the 3D baseline
/*!
 * \brief fft3d_r2c from the home layout with the spectrum left where the last 1D transform puts it.
 */
struct runner_3d{
    runner_3d(std::vector<order3> const &components, heffte::box3d<> const inbox, heffte::box3d<> const outbox,
              heffte::plan_options const options, bool batched, int n, MPI_Comm comm) :
        fft(inbox, outbox, 0, comm, options), k(static_cast<int>(components.size())), batched(batched),
        nr(fft.size_inbox()), nc(fft.size_outbox()),
        fhat(nc), scaled(k * nc), output(k * nr), workspace((batched ? k : 1) * fft.size_workspace())
    {
        double const scale = double(n) * n * n;
        std::vector<std::complex<double>> host_multipliers(k * nc);
        for(int d = 0; d < k; d++){
            int c = 0;
            for_each_index(outbox, [&](std::array<int, 3> const &idx){
                std::complex<double> m = 1.0;
                for(int a = 0; a < 3; a++){
                    int const kk = (a == 0 or 2 * idx[a] <= n) ? idx[a] : idx[a] - n;
                    m *= spectral_factor(kk, n, components[d][a], (a == 0) ? scale : 1.0);
                }
                host_multipliers[d * nc + c++] = m;
            });
        }
        multipliers = to_device(host_multipliers);
    }
    void run(double const input[]){
        fft.forward(input, fhat.data(), workspace.data());
        for(int d = 0; d < k; d++){
            device_multiply(nc, multipliers.data() + d * nc, fhat.data(), scaled.data() + d * nc);
            if (not batched) fft.backward(scaled.data() + d * nc, output.data() + d * nr, workspace.data());
        }
        if (batched) fft.backward(k, scaled.data(), output.data(), workspace.data());
    }
    heffte::fft3d_r2c<backend_tag> fft;
    int k;
    bool batched;
    int nr, nc;
    device_vector<std::complex<double>> fhat, scaled;
    device_vector<double> output;
    device_vector<std::complex<double>> workspace, multipliers;
};

// ------------------------------------------------------------------------------------------- timing
struct timing{ double median, minimum, p90; };

//! \brief Times the call, each run is the slowest rank; one warm-up run is not counted.
template<typename callback>
timing time_runs(callback &&fn, int num_runs, MPI_Comm comm){
    fn();
    device_sync();
    std::vector<double> t(num_runs);
    for(int r = 0; r < num_runs; r++){
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        fn();
        device_sync();
        double local = MPI_Wtime() - t0;
        MPI_Allreduce(&local, &t[r], 1, MPI_DOUBLE, MPI_MAX, comm);
    }
    std::sort(t.begin(), t.end());
    return {t[num_runs / 2], t[0], t[std::min(num_runs - 1, (9 * num_runs) / 10)]};
}

std::string algorithm_name(heffte::reshape_algorithm a){
    switch(a){
        case heffte::reshape_algorithm::alltoallv: return "alltoallv";
        case heffte::reshape_algorithm::alltoall:  return "alltoall";
        case heffte::reshape_algorithm::p2p_plined: return "p2p_plined";
        default: return "p2p";
    }
}

void benchmark_derivatives(int n, int num_runs, MPI_Comm comm){

    int const me        = heffte::mpi::comm_rank(comm);
    int const num_ranks = heffte::mpi::comm_size(comm);
    double const h = 2.0 * 3.14159265358979323846 / n;

    heffte::box3d<> const world({0, 0, 0}, {n - 1, n - 1, n - 1});
    auto const boxes = make_layouts(world, num_ranks);
    std::array<int, 2> const grid = heffte::make_procgrid(num_ranks);

    // the spectrum of the 3D methods: z-pencils for the pencil method, y-split (z-local) for slabs
    heffte::box3d<> const cworld = world.r2c(0);
    heffte::box3d<> const zpencil_spectrum = heffte::split_world(cworld, {grid[0], grid[1], 1})[me];
    heffte::box3d<> const zlocal_spectrum  = heffte::split_world(cworld, {1, num_ranks, 1})[me];

    std::vector<std::pair<std::string, std::vector<order3>>> const sets = {
        {"f_x", {{1, 0, 0}}}, {"f_y", {{0, 1, 0}}}, {"f_z", {{0, 0, 1}}},
        {"f_xx", {{2, 0, 0}}}, {"f_yy", {{0, 2, 0}}}, {"f_zz", {{0, 0, 2}}},
        {"f_xy", {{1, 1, 0}}}, {"f_yz", {{0, 1, 1}}}, {"f_zx", {{1, 0, 1}}}, {"f_xyy", {{1, 2, 0}}},
        {"gradient", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        {"gradient+diag", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}}},
        {"gradient+Hessian", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
                              {1, 1, 0}, {0, 1, 1}, {1, 0, 1}}},
    };

    struct family{
        std::string name;
        int home;
        std::vector<int> layouts;
        heffte::box3d<> spectrum;
        bool use_pencils;
    };
    std::vector<family> const families = {
        {"pencils", X,  {X, Y, Z},    zpencil_spectrum, true},
        {"slabs",   XY, {XY, YZ, XZ}, zlocal_spectrum,  false},
    };

    // f in both home layouts, the exact derivatives are made when a set is checked
    std::map<int, std::vector<double>> input;
    for(auto const &fam : families){
        std::vector<double> &in = input[fam.home];
        for_each_index(boxes[fam.home][me], [&](std::array<int, 3> const &idx){
            double v = 0.0;
            exact_value({0, 0, 0}, idx[0] * h, idx[1] * h, idx[2] * h, v);
            in.push_back(v);
        });
    }
    std::map<int, device_vector<double>> input_device;
    for(auto const &fam : families) input_device[fam.home] = to_device(input[fam.home]);
    auto load_input = [&](int home, double destination[]){
        device_copy(destination, input[home].data(), input[home].size() * sizeof(double));
    };
    auto exact_field = [&](int home, order3 const &c, std::vector<double> &e)->bool{
        e.clear();
        bool known = true;
        for_each_index(boxes[home][me], [&](std::array<int, 3> const &idx){
            double v = 0.0;
            known = exact_value(c, idx[0] * h, idx[1] * h, idx[2] * h, v) and known;
            e.push_back(v);
        });
        return known;
    };
    // relative error against the exact value, or against the 3D result for the third order
    auto error = [&](double const computed[], std::vector<double> const &reference)->double{
        double local[2] = {0.0, 0.0};
        for(size_t i = 0; i < reference.size(); i++){
            local[0] = std::max(local[0], std::abs(reference[i]));
            local[1] = std::max(local[1], std::abs(computed[i] - reference[i]));
        }
        double global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, comm);
        return global[1] / global[0];
    };

    if (me == 0)
        std::cout << "backend " << heffte::backend::name<backend_tag>() << ", grid " << n << "^3, " << num_ranks
                  << " rank(s), pencil grid " << grid[0] << " x " << grid[1] << ", " << num_runs << " timed runs"
#ifdef DERIVATIVES3D_ROCM
                  << (heffte::default_options<backend_tag>().use_gpu_aware ? ", GPU-aware MPI" : ", MPI from the host")
#endif
                  << "\n\n";

    // ------------------------------------------------------------ choosing the options on the full set
    std::vector<order3> const &full = sets.back().second;
    std::vector<heffte::reshape_algorithm> const algorithms = {heffte::reshape_algorithm::alltoallv,
                                                               heffte::reshape_algorithm::alltoall,
                                                               heffte::reshape_algorithm::p2p_plined};
    int const tuning_runs = std::max(3, num_runs / 3);
    struct choice_3d{ heffte::plan_options options = heffte::default_options<backend_tag>(); bool batched = false; std::string name; };
    std::map<std::string, choice_3d> best_3d;
    std::map<std::string, heffte::reshape_algorithm> best_1d;
    for(auto const &fam : families){
        double best = 1.E300;
        for(bool reorder : {true, false}){
            for(auto alg : algorithms){
                for(bool batched : {false, true}){
                    heffte::plan_options options = heffte::default_options<backend_tag>();
                    options.use_pencils = fam.use_pencils;
                    options.use_reorder = reorder;
                    options.algorithm = alg;
                    runner_3d r(full, boxes[fam.home][me], fam.spectrum, options, batched, n, comm);
                    double const t = time_runs([&]{ r.run(input_device[fam.home].data()); }, tuning_runs, comm).median;
                    if (t < best){
                        best = t;
                        best_3d[fam.name] = {options, batched, std::string(reorder ? "reorder" : "no reorder") + " "
                                             + algorithm_name(alg) + (batched ? " batched" : " one at a time")};
                    }
                }
            }
        }
        best = 1.E300;
        plan_1d const plan = best_plan(full, fam.home, fam.layouts);
        for(auto alg : algorithms){
            runner_1d r(plan, boxes, alg, n, comm);
            load_input(fam.home, r.field(fam.home, {0, 0, 0}));
            double const t = time_runs([&]{ r.run(); }, tuning_runs, comm).median;
            if (t < best){
                best = t;
                best_1d[fam.name] = alg;
            }
        }
        if (me == 0) std::cout << "options for " << fam.name << ": 3D " << best_3d[fam.name].name
                               << "; 1D " << algorithm_name(best_1d[fam.name]) << "\n";
    }

    // --------------------------------------------------------------------------------- the campaign
    if (me == 0)
        std::cout << "\nmilliseconds per evaluation, median of " << num_runs << " [min, p90]; speedup of the medians\n\n"
                  << std::left << std::setw(18) << "set" << std::setw(9) << "family" << std::right
                  << std::setw(26) << "3D" << std::setw(26) << "1D" << std::setw(9) << "speedup"
                  << std::setw(8) << "passes" << std::setw(7) << "moved" << std::setw(7) << "calls"
                  << "  " << std::left << std::setw(11) << "route" << std::right << std::setw(9) << "err 1D" << std::setw(9) << "err 3D" << "\n";

    bool pass = true;
    double const tolerance = 1.E-10;
    std::vector<std::string> csv_lines;
    for(auto const &s : sets){
        for(auto const &fam : families){
            auto const &c3 = best_3d[fam.name];
            plan_1d const plan = best_plan(s.second, fam.home, fam.layouts);
            int const k = static_cast<int>(s.second.size());
            double worst = 0.0, worst_3d = 0.0;
            std::vector<double> reference;
            std::map<int, std::vector<double>> from_3d;   // references without an exact formula
            timing t3, t1;
            double split_max[2];
            {   // one method at a time keeps the resident memory down
                runner_3d r3(s.second, boxes[fam.home][me], fam.spectrum, c3.options, c3.batched, n, comm);
                r3.run(input_device[fam.home].data());
                for(int d = 0; d < k; d++){
                    if (exact_field(fam.home, s.second[d], reference))
                        worst_3d = std::max(worst_3d, error(to_host(r3.output.data() + d * r3.nr, r3.nr).data(), reference));
                    else
                        from_3d[d] = to_host(r3.output.data() + d * r3.nr, r3.nr);
                }
                t3 = time_runs([&]{ r3.run(input_device[fam.home].data()); }, num_runs, comm);
            }
            {
                runner_1d r1(plan, boxes, best_1d[fam.name], n, comm);
                load_input(fam.home, r1.field(fam.home, {0, 0, 0}));
                r1.run();
                for(int d = 0; d < k; d++){
                    std::vector<double> const computed = to_host(r1.field(fam.home, s.second[d]), input[fam.home].size());
                    if (from_3d.count(d)) worst = std::max(worst, error(computed.data(), from_3d[d]));
                    else if (exact_field(fam.home, s.second[d], reference))
                        worst = std::max(worst, error(computed.data(), reference));
                }
                r1.time_transform = r1.time_reshape = 0.0;
                t1 = time_runs([&]{ r1.run(); }, num_runs, comm);
                double split[2] = {r1.time_transform / (num_runs + 1), r1.time_reshape / (num_runs + 1)};
                MPI_Allreduce(split, split_max, 2, MPI_DOUBLE, MPI_MAX, comm);
            }
            pass = pass and worst < tolerance and worst_3d < tolerance;

            std::string route = layout_names[plan.home];
            for(int l : plan.route) route += ">" + layout_names[l];
            if (me == 0){
                auto cell = [](timing const &t){
                    std::ostringstream os;
                    os << std::fixed << std::setprecision(2) << 1.E3 * t.median << " [" << 1.E3 * t.minimum << ", " << 1.E3 * t.p90 << "]";
                    return os.str();
                };
                std::ostringstream row;
                row << std::left << std::setw(18) << s.first << std::setw(9) << fam.name << std::right
                    << std::setw(26) << cell(t3) << std::setw(26) << cell(t1)
                    << std::fixed << std::setprecision(2) << std::setw(8) << t3.median / t1.median << "x"
                    << std::setw(8) << plan.passes << std::setw(7) << plan.moved << std::setw(7) << plan.calls
                    << "  " << std::left << std::setw(11) << route << std::right
                    << std::scientific << std::setprecision(1) << std::setw(9) << worst << std::setw(9) << worst_3d;
                std::cout << row.str() << std::endl;
                std::ostringstream csv;
                csv << std::setprecision(6) << "CSV," << n << "," << num_ranks << "," << s.first << "," << fam.name << ","
                    << t3.median << "," << t3.minimum << "," << t3.p90 << ","
                    << t1.median << "," << t1.minimum << "," << t1.p90 << ","
                    << split_max[0] << "," << split_max[1] << ","
                    << plan.passes << "," << plan.moved << "," << plan.calls << "," << route << "," << worst << "," << worst_3d << ","
                    << c3.name << "," << algorithm_name(best_1d[fam.name]);
                csv_lines.push_back(csv.str());
            }
        }
    }
    if (me == 0){
        std::cout << "\n" << (pass ? "all relative errors below " : "FAILED, an error exceeds ") << tolerance << "\n\n"
                  << "CSV,n,ranks,set,family,3d_median,3d_min,3d_p90,1d_median,1d_min,1d_p90,"
                  << "1d_transform,1d_reshape,passes,moved,calls,route,error_1d,error_3d,3d_options,1d_algorithm\n";
        for(auto const &l : csv_lines) std::cout << l << "\n";
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int const n        = (argc > 1) ? std::stoi(argv[1]) : 128;
    int const num_runs = (argc > 2) ? std::stoi(argv[2]) : 20;

    benchmark_derivatives(n, num_runs, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
