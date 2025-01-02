//
// Created by Ethan Shu on 11/25/24.
//


// declaration for the AsianOption base class

class AsianOption {
protected:
	PayOff* pay_off;  // Pay-off class (in this instance call or put)

public:
	AsianOption(PayOff* _pay_off);
	virtual ~AsianOption() {};

	// Pure virtual pay-off operator (this will determine arithmetic or geometric)
	virtual double pay_off_price(const std::vector<double>& spot_prices) const = 0;
};


#ifndef __PAY_OFF_HPP
#define __PAY_OFF_HPP

#include <algorithm> // This is needed for the std::max comparison function, used in the pay-off calculations

class PayOff {
public:
	PayOff(); // Default (no parameter) constructor
	virtual ~PayOff() {}; // Virtual destructor

	// Overloaded () operator, turns the PayOff into an abstract function object
	virtual double operator() (const double& S) const = 0;
};

class PayOffCall : public PayOff {
private:
	double K; // Strike price

public:
	PayOffCall(const double& K_);
	virtual ~PayOffCall() {};

	// Virtual function is now over-ridden (not pure-virtual anymore)
	virtual double operator() (const double& S) const;
};

class PayOffPut : public PayOff {
private:
	double K; // Strike

public:
	PayOffPut(const double& K_);
	virtual ~PayOffPut() {};
	virtual double operator() (const double& S) const;
};

#endif