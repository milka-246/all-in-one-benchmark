#pragma once

#include<omp.h>
#include<iostream>

#include"Benchmark.h"

class ScalarADD : public Benchmark {
public:
	std::string name() const override {
		return "Scalar ADD DP";
	}

	void run() override {
        const unsigned long long ITER = 5000000000ULL;
        const double coeff = 0.123456;

        int num_threads = omp_get_max_threads();
        std::cout << "Threads: " << num_threads << std::endl;

        double start = omp_get_wtime();

        omp_set_num_threads(num_threads);

#pragma omp parallel
        {
            double y0 = 1.0, y1 = 2.0, y2 = 3.0, y3 = 4.0;
            double y4 = 5.0, y5 = 6.0, y6 = 7.0, y7 = 8.0;
            double y8 = 9.0, y9 = 10.0, y10 = 11.0, y11 = 12.0;
            double y12 = 13.0, y13 = 14.0, y14 = 15.0, y15 = 16.0;

            unsigned long long local_iter = ITER / num_threads;

#pragma omp for schedule(static)
            for (int t = 0; t < num_threads; ++t) {
                for (unsigned long long i = 0; i < local_iter; ++i) {
                    y0 += coeff; y1 += coeff;
                    y2 += coeff; y3 += coeff;
                    y4 += coeff; y5 += coeff;
                    y6 += coeff; y7 += coeff;
                    y8 += coeff; y9 += coeff;
                    y10 += coeff; y11 += coeff;
                    y12 += coeff; y13 += coeff;
                    y14 += coeff; y15 += coeff;
                }
            }

            // чтобы не оптимизировало
            volatile double res = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 +
                y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15;
            (void)res;
        }

        double stop = omp_get_wtime();

        unsigned long long total_flop = ITER * 16ULL;
        double perf = total_flop / ((stop - start) * 1e9);

        std::cout << "Time: " << (stop - start) << " s" << std::endl;
        std::cout << "GFLOPS: " << perf << std::endl;
	}
};