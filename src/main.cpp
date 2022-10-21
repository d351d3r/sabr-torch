#include <torch/torch.h>
#include <iostream>


// TODO Did't work
#include "LevenbergMarquad.hpp"

#include "sigma_SABR.hpp"


// TODO Input function
static double f(int t, int p) {
    return p[0] * std::exp(-t / p[1]) + p[2] * t * std::exp(-t / p[3]);
}


static double K = torch::linspace(0.04, 0.11, 25),
S = 0.06,
T = 0.5,
alpha = 0.037561,
beta = 0.5,
rho = 0.100044,
nu = 0.573296;

//sigma_SABR
//LevenbergMarquad lm();
int main() {
    LevenbergMarquad(1,1,1);
    auto true_p = torch::Tensor([20.0, 10.0, 1.0, 50.0]); // True parameteres
    auto x_true = torch::linspace(0, 100, 25); // span of of free parameter
    auto y_true = f(x_true, true_p); // fitted function observed values

// TODO
// init_p = true_p + torch.randn(4) * 2 ** 2 + 4  # initial guess for parametes = true + noise
//modified_f = data_points(x_true)(f)  # wrapped function (dependent on only parameters)
//    std::cout << "Initial guess for optimizer " << init_p << std::endl;
}