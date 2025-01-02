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
	// 5 factors affecting option prices
	double currentStockPrice;
	double strike;
	double maturity;
	double riskFreeRate;
	double volatility;
	// simulation params
	int numSteps;
	int numPaths;
	// types of asian option
	bool isCall;
	bool useArithmetic;

public:
	// default constructor
	AsianOption(double currentStockPrice, double strike, double maturity, double riskFreeRate,
	            double volatility, int numSteps, int numPaths, bool isCall, bool useArithmetic)
			: currentStockPrice(currentStockPrice), strike(strike), maturity(maturity),
			  riskFreeRate(riskFreeRate), volatility(volatility), numSteps(numSteps),
			  numPaths(numPaths), isCall(isCall), useArithmetic(useArithmetic) {}
	// define purely virtual function and the function doesn't modify the state of the object on which it is called
	virtual double simulatePayoff(std::mt19937 &rng) const = 0;


	double price() const {
		///
		/// member function
		///

		// variable -- the starting time
		auto start = std::chrono::high_resolution_clock::now();
		// number of hardware cores for parallel computing
		const int numThreads = std::thread::hardware_concurrency();
		// a vector of threads called threads
		std::vector<std::thread> threads;
		// constructor of vector: the initial size of it is numThreads; the initial values of those elements is 0.
		std::vector<double> results(numThreads, 0.0);
		// lambda function: parallel Monte Carlo simulation worker function for calculating payoffs
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
	AsianOptionMonteCarlo(double currentStockPrice, double strike, double maturity, double riskFreeRate,
	                      double volatility, int numSteps, int numPaths, bool isCall, bool useArithmetic)
			: AsianOption(currentStockPrice, strike, maturity, riskFreeRate, volatility,
			              numSteps, numPaths, isCall, useArithmetic) {}

	double simulatePayoff(std::mt19937 &rng) const override {
		std::normal_distribution<double> dist(0.0, 1.0);
		double dt = maturity / numSteps;
		double drift = exp((riskFreeRate - 0.5 * volatility * volatility) * dt);
		double diffusion = exp(volatility * sqrt(dt));
		std::vector<double> prices(numSteps + 1, currentStockPrice);

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
	// User inputs
	double currentStockPrice = 100;
	double strike = 100;
	double maturity = 2 ;
	double riskFreeRate = 0.05;
	double volatility = 0.2;
	int numSteps = 252; // Default number of steps
	int numPaths = 1000000; // Default number of paths
	bool isCall = true;
	bool useArithmetic = true;

	/*
	std::cout << "Enter current stock price: ";
	std::cin >> currentStockPrice;
	std::cout << "Enter strike price: ";
	std::cin >> strike;
	std::cout << "Enter maturity (in years): ";
	std::cin >> maturity;
	std::cout << "Enter risk-free rate: ";
	std::cin >> riskFreeRate;
	std::cout << "Enter volatility (as a decimal, e.g., 0.2 for 20%): ";
	std::cin >> volatility;
	std::cout << "Enter 1 for Call option, 0 for Put option: ";
	std::cin >> isCall;
	std::cout << "Enter 1 for Arithmetic average, 0 for Geometric average: ";
	std::cin >> useArithmetic;
	 */

	AsianOptionMonteCarlo option(currentStockPrice, strike, maturity, riskFreeRate,
	                             volatility, numSteps, numPaths, isCall, useArithmetic);
	std::cout << "Asian Option Price: " << option.price() << std::endl;

	return 0;
}
