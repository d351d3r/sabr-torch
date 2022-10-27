#include <torch/torch.h>
#include <iostream>
#include <list>
#include <vector>

// TODO Did't work
#include "LevenbergMarquad.hpp"
#include "sigma_SABR.hpp"


// TODO Input function
// Need convert  return p[0] * torch.exp(-t / p[1]) + p[2] * t * torch.exp(-t / p[3])

static double f(int t, std::vector p) {
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
    auto true_p = torch::Tensor([20.0, 10.0, 1.0, 50.0]); // True parameteres
    auto x_true = torch::linspace(0, 100, 25); // span of of free parameter
    auto y_true = f(x_true, true_p); // fitted function observed values

    // TODO
    auto init_p = true_p + pow((torch::randn(4) * 2), 2) + 4;
/// // initial guess for parametes = true + noise
    auto modified_f = func(x_true, f);  // wrapped function (dependent on only parameters)
    //    std::cout << "Initial guess for optimizer " << init_p << std::endl;


    LevenbergMarquad lm(init_p,x_true, init_p, y_true, modified_f);
    for (int i = 0; i <= 1000; ++i)
        lm.step();

    std::list<float> p_hat;
    std::list<float> p;
    std::list<float> lmbd;

    for (int i = 0; i < 10; i++) {
        p.push_back(float(torch::linalg::norm(lm.step(), ord = 2)));
        p_hat.push_back(float(torch::linalg::norm(lm.p - true_p, ord = 2)));
        lmbd.push_back(lm.lambda_lm);
    }

//TODO: Sigma
    auto K = torch::linspace(0.04, 0.11, 25);
    double S = 0.06;
    double T = 0.5;
    double alpha = 0.037561;
    double beta = 0.5;
    double rho = 0.100044;
    double nu = 0.573296;

    init_p = torch::tensor([0.1] * 3);
    true_p = torch::tensor([alpha, nu, rho]);
    auto y_mkt = sigma_SABR(K, true_p);

    modified_sabr = func(K, sigma_SABR);


    LevenbergMarquad lm_sigma(x_true, init_p, y_true, modified_f);
    for (int i = 0; i <= 1000; ++i)
        lm.step();

}