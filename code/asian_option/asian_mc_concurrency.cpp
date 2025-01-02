//
// Created by Ethan Shu on 12/1/24.
//

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <numeric>
#include <functional>

class AsianOption {
protected:
	double strike;
	double maturity;
	double riskFreeRate;
	double volatility;
	int numSteps;
	int numPaths;
	bool isCall;
	bool useArithmetic;

public:
	AsianOption(double strike, double maturity, double riskFreeRate, double volatility,
	            int numSteps, int numPaths, bool isCall, bool useArithmetic)
			: strike(strike), maturity(maturity), riskFreeRate(riskFreeRate),
			  volatility(volatility), numSteps(numSteps), numPaths(numPaths),
			  isCall(isCall), useArithmetic(useArithmetic) {}

	virtual double simulatePayoff(std::mt19937 &rng) const = 0;

	double price() const {
		auto start = std::chrono::high_resolution_clock::now();

		const int numThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		std::vector<double> results(numThreads, 0.0);

		auto worker = [&](int threadId) {
			std::mt19937 rng(threadId); // Seed with thread ID
			double sumPayoffs = 0.0;

			int pathsPerThread = numPaths / numThreads;
			for (int i = 0; i < pathsPerThread; ++i) {
				sumPayoffs += simulatePayoff(rng);
			}

			results[threadId] = sumPayoffs;
		};

		for (int t = 0; t < numThreads; ++t) {
			threads.emplace_back(worker, t);
		}

		for (auto &thread : threads) {
			thread.join();
		}

		double totalPayoff = std::accumulate(results.begin(), results.end(), 0.0);
		double discountedPayoff = (totalPayoff / numPaths) * exp(-riskFreeRate * maturity);

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Run Time: " << elapsed.count() << " seconds" << std::endl;

		return discountedPayoff;
	}
};

class AsianOptionMonteCarlo : public AsianOption {
public:
	AsianOptionMonteCarlo(double strike, double maturity, double riskFreeRate, double volatility,
	                      int numSteps, int numPaths, bool isCall, bool useArithmetic)
			: AsianOption(strike, maturity, riskFreeRate, volatility, numSteps, numPaths, isCall, useArithmetic) {}

	double simulatePayoff(std::mt19937 &rng) const override {
		std::normal_distribution<double> dist(0.0, 1.0);
		double dt = maturity / numSteps;
		double drift = exp((riskFreeRate - 0.5 * volatility * volatility) * dt);
		double diffusion = exp(volatility * sqrt(dt));
		std::vector<double> prices(numSteps + 1, 100.0); // Assume initial price is 100

		for (int i = 1; i <= numSteps; ++i) {
			double z = dist(rng);
			prices[i] = prices[i - 1] * drift * pow(diffusion, z);
		}

		double avgPrice = useArithmetic
		                  ? std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size()
		                  : exp(std::accumulate(prices.begin(), prices.end(), 0.0,
		                                        [](double sum, double p) { return sum + log(p); }) /
		                        prices.size());

		double payoff = isCall ? std::max(avgPrice - strike, 0.0) : std::max(strike - avgPrice, 0.0);
		return payoff;
	}
};

int main() {
	double strike = 100.0;
	double maturity = 2.0;
	double riskFreeRate = 0.05;
	double volatility = 0.2;
	int numSteps = 504;
	int numPaths = 5000;
	bool isCall = true;       // User can choose true for call, false for put
	bool useArithmetic = true; // User can choose true for arithmetic, false for geometric

	AsianOptionMonteCarlo option(strike, maturity, riskFreeRate, volatility, numSteps, numPaths, isCall, useArithmetic);

	std::cout << "Asian Option Price: " << option.price() << std::endl;

	return 0;
}
