/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include <random>

#include "test_common.h"

void test_factorize(){
    current_test<int, using_nompi> name("prime factorize");

    std::vector<std::array<int, 2>> reference = {{1, 935}, {5, 187}, {11, 85}, {17, 55}, {55, 17}, {85, 11}, {187, 5}};

    auto factors = heffte::get_factors(935);

    sassert(match(factors, reference));

    reference = {{1, 27}, {3, 9}, {9, 3}};
    factors = heffte::get_factors(reference.front()[1]);
    sassert(match(factors, reference));
}

void test_process_grid(){
    current_test<int, using_nompi> name("process grid");

    std::array<int, 2> reference = {4, 5};
    std::array<int, 2> result = heffte::make_procgrid(20);
    sassert(reference == result);

    reference = {1, 17};
    result = heffte::make_procgrid(17);
    sassert(reference == result);

    reference = {81, 81};
    result = heffte::make_procgrid(6561);
    sassert(reference == result);

    reference = {17, 19};
    result = heffte::make_procgrid(323);
    sassert(reference == result);

    reference = {8, 16};
    result = heffte::make_procgrid(128);
    sassert(reference == result);
}

void test_split_pencils(){
    using namespace heffte;
    current_test<int, using_nompi> name("split pencils");

    box3d const world = {{0, 0, 0}, {1, 3, 5}};
    std::vector<box3d> reference = {{{0, 0, 0}, {0, 1, 5}}, {{0, 2, 0}, {0, 3, 5}},
                                    {{1, 0, 0}, {1, 1, 5}}, {{1, 2, 0}, {1, 3, 5}}};
    // note that the order of the boxes moves fastest in the mid-dimension
    // this tests the reordering
    std::vector<box3d> result = make_pencils(world, {2, 2}, 2, reference);
    sassert(match(result, reference));

    std::vector<box3d> reference2 = {{{0, 0, 0}, {1, 1, 2}}, {{0, 2, 0}, {1, 3, 2}},
                                     {{0, 0, 3}, {1, 1, 5}}, {{0, 2, 3}, {1, 3, 5}}};
    std::vector<box3d> result2 = make_pencils(world, {2, 2}, 0, reference);
    sassert(match(result2, reference2));

    box3d const reconstructed_world = find_world(result);
    sassert(reconstructed_world == world);
}

void test_cpu_scale(){
    using namespace heffte;
    current_test<int, using_nompi> name("cpu scaling");
    std::vector<float> x = {1.0, 33.0, 88.0, -11.0, 2.0};
    std::vector<float> y = x;
    for(auto &v : y) v *= 3.0;
    data_scaling<tag::cpu>::apply(x.size(), x.data(), 3.0);
    sassert(approx(x, y));

    std::vector<std::complex<double>> cx = {{1.0, -11.0}, {33.0, 8.0}, {88.0, -11.0}, {2.0, -9.0}};
    std::vector<std::complex<double>> cy = cx;
    for(auto &v : cy) v /= 1.33;
    data_scaling<tag::cpu>::apply(cx.size(), cx.data(), 1.0 / 1.33);
    sassert(approx(cx, cy));
}

/*
 * Generates input for the fft, the input consists of reals or complex but they have only
 * integer values and the values follow the order of the entries.
 * Designed to work on a grid of size {2, 3, 4} for total of 24 entries.
 */
template<typename scalar_type>
std::vector<scalar_type> make_input(){
    std::vector<scalar_type> result(24);
    for(int i=0; i<24; i++) result[i] = static_cast<scalar_type>(i + 1);
    return result;
}

/*
 * Considering the input generated by the make_input() method and box {{0, 0, 0}, {1, 2, 3}}
 * constructs the corresponding fft coefficients assuming 1-D transforms have
 * been applied across the zeroth dimension.
 * Each transform uses 2 entries, since the size in dimension 0 is 2.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft0(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<result.size(); i+=2){
        result[i]   = static_cast<typename fft_output<scalar_type>::type>(3 + 2 * i);
        result[i+1] = static_cast<typename fft_output<scalar_type>::type>(-1.0);
    }
    return result;
}
/*
 * Same as make_fft0() but the transforms are applied to dimension 1.
 * Each transform uses 3 entries, since the size in dimension 1 is 3.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft1(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(int j=0; j<4; j++){
        for(int i=0; i<2; i++){
            result[6 * j + i]     = typename fft_output<scalar_type>::type((2*j + i+1) * 9.0 - i * 6.0);
            result[6 * j + i + 2] = typename fft_output<scalar_type>::type(-3.0,  1.73205080756888);
            result[6 * j + i + 4] = typename fft_output<scalar_type>::type(-3.0, -1.73205080756888);
        }
    }
    return result;
}
/*
 * Same as make_fft1() but using the r2c transform and only the unique entries.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft1_r2c(){
    std::vector<typename fft_output<scalar_type>::type> result(16);
    for(int j=0; j<4; j++){
        for(int i=0; i<2; i++){
            result[4 * j + i]     = typename fft_output<scalar_type>::type((2*j + i+1) * 9.0 - i * 6.0);
            result[4 * j + i + 2] = typename fft_output<scalar_type>::type(-3.0,  1.73205080756888);
        }
    }
    return result;
}
/*
 * Same as make_fft0() but the transforms are applied to dimension 2.
 * Each transform uses 4 entries, since the size in dimension 2 is 4.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft2(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<6; i++){
        result[i]    = typename fft_output<scalar_type>::type(40.0 + 4 * i);
        result[i+ 6] = typename fft_output<scalar_type>::type(-12.0, 12.0);
        result[i+12] = typename fft_output<scalar_type>::type(-12.0);
        result[i+18] = typename fft_output<scalar_type>::type(-12.0, -12.0);
    }
    return result;
}
/*
 * Same as make_fft2() but using the r2c transform and only the unique entries.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft2_r2c(){
    std::vector<typename fft_output<scalar_type>::type> result(18);
    for(size_t i=0; i<6; i++){
        result[i]    = typename fft_output<scalar_type>::type(40.0 + 4 * i);
        result[i+ 6] = typename fft_output<scalar_type>::type(-12.0, 12.0);
        result[i+12] = typename fft_output<scalar_type>::type(-12.0);
    }
    return result;
}

#ifdef Heffte_ENABLE_FFTW
template<typename scalar_type>
void test_fftw_1d_complex(){
    current_test<scalar_type, using_nompi> name("fftw3 one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with make_input and make_fft methods

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::fftw_executor fft(box, i);

        std::vector<scalar_type> result = input;
        fft.forward(result.data());
        sassert(approx(result, reference[i]));

        fft.backward(result.data());
        for(auto &r : result) r /= (2.0 + i);
        sassert(approx(result, input));
    }
}

template<typename scalar_type>
void test_fftw_1d_real(){
    current_test<scalar_type, using_nompi> name("fftw3 one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::fftw_executor fft(box, i);

        std::vector<typename fft_output<scalar_type>::type> result(input.size());
        fft.forward(input.data(), result.data());
        sassert(approx(result, reference[i]));

        std::vector<scalar_type> back_result(result.size());
        fft.backward(result.data(), back_result.data());
        for(auto &r : back_result) r /= (2.0 + i);
        sassert(approx(back_result, input));
    }
}

template<typename scalar_type>
void test_fftw_1d_r2c(){
    current_test<scalar_type, using_nompi> name("fftw3 one-dimension r2c");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1_r2c<scalar_type>(), make_fft2_r2c<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::fftw_executor_r2c fft(box, i);

        std::vector<typename fft_output<scalar_type>::type> result(fft.complex_size());
        fft.forward(input.data(), result.data());
        sassert(approx(result, reference[i]));

        std::vector<scalar_type> back_result(fft.real_size());
        fft.backward(result.data(), back_result.data());
        for(auto &r : back_result) r /= (2.0 + i);
        sassert(approx(back_result, input));
    }
}

// tests for the 1D fft
void test_fftw(){
    test_fftw_1d_real<float>();
    test_fftw_1d_real<double>();
    test_fftw_1d_complex<std::complex<float>>();
    test_fftw_1d_complex<std::complex<double>>();
    test_fftw_1d_r2c<float>();
    test_fftw_1d_r2c<double>();
}
#else
void test_fftw(){}
#endif

#ifdef Heffte_ENABLE_CUDA
template<typename scalar_type>
void test_cuda_vector_type(size_t num_entries){
    current_test<scalar_type, using_nompi> name("cuda::vector");
    std::vector<scalar_type> source(num_entries);
    std::iota(source.begin(), source.end(), 0); // fill source with 0, 1, 2, 3, 4 ...
    cuda::vector<scalar_type> v1 = cuda::load(source);
    sassert(v1.size() == source.size());
    cuda::vector<scalar_type> v2 = v1; // test copy constructor
    sassert(v1.size() == v2.size());
    std::vector<scalar_type> dest = cuda::unload(v2);
    sassert(match(dest, source));

    { // test move constructor
        cuda::vector<scalar_type> t = std::move(v2);
        dest = std::vector<scalar_type>(); // reset the destination
        dest = cuda::unload(t);
        sassert(match(dest, source));
    }

    sassert(v2.empty()); // test empty and reset to null after move
    v2 = std::move(v1);  // test move assignment
    sassert(v1.empty()); // test if moved out of v1

    dest = std::vector<scalar_type>(); // reset the destination
    dest = cuda::unload(v2);
    sassert(match(dest, source));

    v1 = cuda::load(source);
    v2 = cuda::vector<scalar_type>(v1.data(), v1.data() + num_entries / 2);
    sassert(v2.size() == num_entries / 2);
    dest = cuda::unload(v2);
    source.resize(num_entries / 2);
    sassert(match(dest, source));

    size_t num_v2 = v2.size();
    scalar_type *raw_array = v2.release();
    sassert(v2.empty());
    v2 = cuda::capture(std::move(raw_array), num_v2);
    sassert(raw_array == nullptr);
    sassert(not v2.empty());
}

void test_cuda_vector(){
    test_cuda_vector_type<float>(11);
    test_cuda_vector_type<double>(40);
    test_cuda_vector_type<std::complex<float>>(73);
    test_cuda_vector_type<std::complex<double>>(13);
}
void test_gpu_scale(){
    using namespace heffte;
    current_test<int, using_nompi> name("cpu scaling");
    std::vector<float> x = {1.0, 33.0, 88.0, -11.0, 2.0};
    std::vector<float> y = x;
    for(auto &v : y) v *= 3.0;
    auto gx = cuda::load(x);
    data_scaling<tag::gpu>::apply(gx.size(), gx.data(), 3.0);
    x = cuda::unload(gx);
    sassert(approx(x, y));

    std::vector<std::complex<double>> cx = {{1.0, -11.0}, {33.0, 8.0}, {88.0, -11.0}, {2.0, -9.0}};
    std::vector<std::complex<double>> cy = cx;
    for(auto &v : cy) v /= 1.33;
    auto gcx = cuda::load(cx);
    data_scaling<tag::gpu>::apply(gcx.size(), gcx.data(), 1.0 / 1.33);
    cx = cuda::unload(gcx);
    sassert(approx(cx, cy));
}
template<typename scalar_type>
void test_cufft_1d_complex(){
    current_test<scalar_type, using_nompi> name("cufft one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with make_input and make_fft methods

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::cufft_executor fft(box, i);

        auto curesult = cuda::load(input);
        fft.forward(curesult.data());
        sassert(approx(curesult, reference[i]));

        fft.backward(curesult.data());
        auto result = cuda::unload(curesult);
        for(auto &r : result) r /= (2.0 + i);
        sassert(approx(result, input));
    }
}
template<typename scalar_type>
void test_cufft_1d_real(){
    current_test<scalar_type, using_nompi> name("cufft one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::cufft_executor fft(box, i);

        cuda::vector<typename fft_output<scalar_type>::type> curesult(input.size());
        auto cuinput = cuda::load(input);
        fft.forward(cuinput.data(), curesult.data());
        sassert(approx(curesult, reference[i]));

        cuda::vector<scalar_type> cuback_result(curesult.size());
        fft.backward(curesult.data(), cuback_result.data());
        auto back_result = cuda::unload(cuback_result);
        for(auto &r : back_result) r /= (2.0 + i);
        sassert(approx(back_result, input));
    }
}

template<typename scalar_type>
void test_cufft_1d_r2c(){
    current_test<scalar_type, using_nompi> name("cufft one-dimension r2c");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with make_input and make_fft methods

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1_r2c<scalar_type>(), make_fft2_r2c<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::cufft_executor_r2c fft(box, i);

        cuda::vector<typename fft_output<scalar_type>::type> result(fft.complex_size());
        auto cuinput = cuda::load(input);
        fft.forward(cuinput.data(), result.data());
        sassert(approx(result, reference[i], 0.01));

        cuda::vector<scalar_type> back_result(fft.real_size());
        fft.backward(result.data(), back_result.data());
        data_scaling<tag::gpu>::apply(back_result.size(), back_result.data(), 1.0 / (2.0 + i));
        sassert(approx(back_result, input, 0.01));
    }
}

void test_cufft(){
    test_cufft_1d_real<float>();
    test_cufft_1d_real<double>();
    test_cufft_1d_complex<std::complex<float>>();
    test_cufft_1d_complex<std::complex<double>>();
    test_cufft_1d_r2c<float>();
    test_cufft_1d_r2c<double>();
}
#else
void test_cuda_vector(){} // skip cuda
void test_gpu_scale(){}
void test_cufft(){}
#endif

#if defined(Heffte_ENABLE_FFTW) and defined(Heffte_ENABLE_CUDA)
template<typename scalar_type>
std::vector<scalar_type> make_data(box3d const world){
    std::minstd_rand park_miller(4242);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<scalar_type> result(world.count());
    for(auto &r : result)
        r = static_cast<scalar_type>(unif(park_miller));
    return result;
}
template<typename scalar_type>
void test_cross_reference_type(){
    current_test<scalar_type, using_nompi> name("cufft - fftw reference");

    box3d box = {{0, 0, 0}, {42, 75, 23}};
    auto input = make_data<scalar_type>(box);
    auto cuinput = cuda::load(input);

    for(int i=0; i<3; i++){
        heffte::fftw_executor  fft_cpu(box, i);
        heffte::cufft_executor fft_gpu(box, i);

        fft_cpu.forward(input.data());
        fft_gpu.forward(cuinput.data());

        if (std::is_same<scalar_type, std::complex<float>>::value){
            sassert(approx(cuinput, input, 0.0005)); // float complex is not well conditioned
        }else{
            sassert(approx(cuinput, input));
        }
    }
}
template<typename scalar_type>
void test_cross_reference_r2c(){
    current_test<scalar_type, using_nompi> name("cufft - fftw reference r2c");

    for(int case_counter = 0; case_counter < 2; case_counter++){
        // due to alignment issues on the cufft side
        // need to check the case when both size[0] and size[1] are odd
        //                        when at least one is even
        box3d box = (case_counter == 0) ?
                    box3d({0, 0, 0}, {42, 70, 21}) :
                    box3d({0, 0, 0}, {41, 50, 21});

        auto input = make_data<scalar_type>(box);
        cuda::vector<scalar_type> cuinput = cuda::load(input);

        for(int i=0; i<3; i++){
            heffte::fftw_executor_r2c  fft_cpu(box, i);
            heffte::cufft_executor_r2c fft_gpu(box, i);

            std::vector<typename fft_output<scalar_type>::type> result(fft_cpu.complex_size());
            cuda::vector<typename fft_output<scalar_type>::type> curesult(fft_gpu.complex_size());

            fft_cpu.forward(input.data(), result.data());
            fft_gpu.forward(cuinput.data(), curesult.data());

            if (std::is_same<scalar_type, float>::value){
                sassert(approx(curesult, result, 0.0005)); // float complex is not well conditioned
            }else{
                sassert(approx(curesult, result));
            }

            std::vector<scalar_type> inverse(fft_cpu.real_size());
            cuda::vector<scalar_type> cuinverse(fft_gpu.real_size());

            fft_cpu.backward(result.data(), inverse.data());
            fft_gpu.backward(curesult.data(), cuinverse.data());

            data_scaling<tag::cpu>::apply(inverse.size(), inverse.data(), 1.0 / static_cast<double>(box.size[i]));
            data_scaling<tag::gpu>::apply(cuinverse.size(), cuinverse.data(), 1.0 / static_cast<double>(box.size[i]));

            if (std::is_same<scalar_type, float>::value){
                sassert(approx(inverse, input));
                sassert(approx(cuinverse, input));
            }else{
                sassert(approx(inverse, input));
                sassert(approx(cuinverse, input));
            }
        }
    }
}
void test_cross_reference(){
    test_cross_reference_type<std::complex<float>>();
    test_cross_reference_type<std::complex<double>>();
    test_cross_reference_r2c<float>();
    test_cross_reference_r2c<double>();
}
#else
void test_cross_reference(){}
#endif

int main(int argc, char *argv[]){

    all_tests<using_nompi> name("Non-MPI Tests");

    test_factorize();
    test_process_grid();
    test_split_pencils();
    test_cpu_scale();

    test_cuda_vector();
    test_gpu_scale();

    test_fftw();
    test_cufft();

    test_cross_reference();

    return 0;
}
