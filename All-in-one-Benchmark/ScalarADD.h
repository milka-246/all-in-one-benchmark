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

        double y0 = 1.0;
        double y1 = 2.0;
        double y2 = 3.0;
        double y3 = 4.0;
        double y4 = 5.0;
        double y5 = 6.0;
        double y6 = 7.0;
        double y7 = 8.0;
        double y8 = 9.0;
        double y9 = 10.0;

        const double coeff = 0.123456;

        double start = omp_get_wtime();

#pragma omp parallel for
#pragma novector
        for (unsigned long long i = 0; i < ITER; ++i) {
            y0 = y0 + coeff;
            y1 = y1 + coeff;
            y2 = y2 + coeff;
            y3 = y3 + coeff;
            y4 = y4 + coeff;
            y5 = y5 + coeff;
            y6 = y6 + coeff;
            y7 = y7 + coeff;
            y8 = y8 + coeff;
            y9 = y9 + coeff;
        }

        double stop = omp_get_wtime();

        // чтобы не выкинуло оптимизатор
        volatile double res = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9;
        std::cout << "Result checksum: " << res << std::endl;

        unsigned long long total_flop = ITER * 10ULL;
        double perf = total_flop / ((stop - start) * 1e9);

        std::cout << "Time: " << (stop - start) << " s" << std::endl;
        std::cout << "GFLOPS: " << perf << std::endl;
	}
};