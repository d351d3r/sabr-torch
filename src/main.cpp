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
    LevenbergMarquad(1,1,1);
    auto true_p = torch::Tensor([20.0, 10.0, 1.0, 50.0]); // True parameteres
    auto x_true = torch::linspace(0, 100, 25); // span of of free parameter
    auto y_true = f(x_true, true_p); // fitted function observed values

    // TODO
    auto init_p = true_p + pow((torch::randn(4) * 2), 2) + 4;
/// // initial guess for parametes = true + noise
    auto modified_f = func(x_true,f);  // wrapped function (dependent on only parameters)
    //    std::cout << "Initial guess for optimizer " << init_p << std::endl;

//TODO
    std::map<std::string,std::string> params = {'x': x_true,
          'init_p': init_p,
          'y': y_true,
          'func': modified_f,  # model to be fitted
          'sigma': 4.0,  # std of datapoint
          'lambda_lm': 10,  # starting lambda
          'eps1': 1e-3,
          'eps2': 1e-3,
          'eps3': 1e-3,
          'eps4': 1e-3,
          'lm_up': 11,  # up coefficient
          'lm_down': 9  # down coefficient
          }
    lm = LevenbergMarquad(params)
            for (int i = 0; i<=1000;++i)
                lm.step();
//p_hat = []
//p = []
//lmbd = []
//for i in range(10):
//    p.append(float(torch.linalg.norm(lm.step(), ord=2)))
//    p_hat.append(float(torch.linalg.norm(lm.p - true_p, ord=2)))
//    lmbd.append(lm.lambda_lm)


//TODO: Sigma


}