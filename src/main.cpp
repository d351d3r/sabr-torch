//#include <torch/torch.h>
//#include <fstream>
//#include <iostream>
//#include <list>
//#include <vector>
//
//#include "LevenbergMarquad.hpp"
//#include "sigma_SABR.hpp"
//
//template <class Data>
//auto data_points(Data data) {
//  auto middle{[data](auto func) {
//    return [data, func](auto... args) { return func(data, args...); };
//  }};
//
//  return middle;
//}
//
//static torch::Tensor f(torch::Tensor t, torch::Tensor p) {
//  return p[0] * exp(-t / p[1]) + p[2] * t * exp(-t / p[3]);
//}
//
//struct LM_PARAMS {
//  torch::Tensor x;
//  torch::Tensor init_p;
//  torch::Tensor y;
//  torch::Tensor func;
//  float sigma = 4.0; // std of datapoint
//  int lambda_lm = 10; // starting lambda
//  float eps1 = 1e-3;
//  float eps2 = 1e-3;
//  float eps3 = 1e-3;
//  float eps4 = 1e-3;
//  int lm_up = 11; // up coefficient
//  int lm_down = 9; // down coefficient
//};
//
//int main() {
//  std::fstream strm;
//  strm.open("input.txt", std::ios_base::out);
//
//  if (strm.is_open()) {
//    std::string str;
//    strm << str;
//    strm.close();
//  } else
//    std::cerr << "File could not be opened!" << std::endl;
//
//  // TODO Input from file (matrix)
//  // striike,val,sigm
//
//  torch::Tensor true_p = torch::randn({20, 10, 1, 50}); // True parameteres
//  torch::Tensor x_true = torch::linspace(0, 100, 25); // span of free parameter
//  torch::Tensor y_true = f(x_true, true_p); // fitted function observed values
//
//  torch::Tensor init_p = true_p + pow((torch::randn(4) * 2), 2) + 4;
//  /// // initial guess for parametes = true + noise
//  auto modified_f =
//      data_points(x_true)(f); // wrapped function (dependent on only parameters)
//  std::cout << "Initial guess for optimizer " << init_p << std::endl;
//
//  //     LevenbergMarquad lm(params)
//  torch::Tensor f;
//  LM_PARAMS lm_params{};
//  LevenbergMarquad lm(
//      init_p,
//      x_true,
//      y_true,
//      modified_f,
//      lm_params.sigma,
//      lm_params.lambda_lm,
//      lm_params.eps1,
//      lm_params.eps2,
//      lm_params.eps3,
//      lm_params.eps4);
//
//  for (int i = 0; i <= 1000; ++i)
//    lm.step();
//
//  std::list<float> p_hat;
//  std::list<float> p;
//  std::list<float> lmbd;
//
//  for (int i = 0; i < 10; i++) {
//    p.push_back(float(torch::linalg::norm(lm.step(), 2)));
//    p_hat.push_back(
//        float(torch::linalg::norm(lm.p - true_p, auto opt_ord = 2)));
//    lmbd.push_back(lm.lambda_lm);
//  }
//
//  // TODO: Sigma
//  torch::Tensor K = torch::linspace(0.04, 0.11, 25);
//  float S = 0.06;
//  float T = 0.5;
//  float alpha = 0.037561;
//  float beta = 0.5;
//  float rho = 0.100044;
//  float nu = 0.573296;
//
//  init_p = torch::randn({0.1 * 3});
//  true_p = torch::randn([ alpha, nu, rho ]);
//    torch::Tensor(y_mkt = sigma_SABR(K, true_p);
//
//
//// TODO
//    auto modified_sabr = data_points(K),(sigma_SABR);
//// TODO
//    struct SIGMA_PARAMS : LM_PARAMS {
//    modified_sabr(alpha, nu, rho);
//    func = modified_sabr;
//
//    float sigma = 1.0; // std of datapoint
//    int lambda_lm = 0.1; // starting lambda
//    int lm_up = 11; // up coefficient
//    int lm_down = 9; // down coefficient
//};
//
//    // TODO
//    LevenbergMarquad lm_sigma(x_true, init_p, y_true, modified_f);
//    for (int i = 0; i <= 1000; ++i)
//        lm.step();
//}
#include <torch/torch.h>
#include <cmath>

class Function {
public:
    static torch::Tensor f(const torch::Tensor& t, const torch::Tensor& p) {
        return p[0] * torch::exp(-t / p[1]) + p[2] * t * torch::exp(-t / p[3]);
    }
};

class LevenbergMarquad {
public:
    LevenbergMarquad(const torch::Tensor& y, const torch::Tensor& x, const torch::Tensor& init_p, const double& sigma, const double& lambda_lm,
                     const double& eps1, const double& eps2, const double& eps3, const double& eps4, const double& lm_up, const double& lm_down)
            : y_data(y), x_data(x), p(init_p), lambda_lm(lambda_lm), eps1(eps1), eps2(eps2), eps3(eps3), eps4(eps4), lm_up(lm_up), lm_down(lm_down)
    {
        p.set_requires_grad(true);
        J = torch::autograd::functional::jacobian(Function::f, p);
        W = torch::diag(torch::ones({ static_cast<int64_t>(x_data.size(0)) }) / (sigma * sigma));
    }

    void broyden_jacobian_update() {
        auto df = Function::f(p + dp, x_data) - Function::f(p, x_data);
        J += torch::outer(df - J.mv(dp), dp) / dp.norm(2);
    }

    void torch_jacobian_update() {
        J = torch::autograd::functional::jacobian(Function::f, p);
    }

    void solve_for_dp() {
        JTW = torch::mm(J.transpose(0, 1), W);
        JTWJ = torch::mm(JTW, J);

        auto dy = y_data - Function::f(p, x_data);
        dp = torch::solve(torch::mv(JTW, dy), JTWJ + lambda_lm * torch::diag(JTWJ.diag()))[0];
    }

    double chi_2() {
        auto y_hat = Function::f(p, x_data);
        return torch::dot(y_data, torch::mv(W, y_data))
               - 2 * torch::dot(y_data, torch::mv(W, y_hat))
               + torch::dot(y_hat, torch::mv(W, y_hat)).item<double>();
    }

    bool rho() {
        auto dy = y_data - Function::f(p, x_data);
        auto rho = (chi_2() - chi_2(p + dp))
                   / torch::dot(dp, lambda_lm * torch::diag(JTWJ.diag()) * dp + torch::mv(JTW, dy)).item<double>();

        return rho > eps4;
    }

    void update_p() {
        p = p + dp;
    }

    void step() {
        dp = torch::zeros({ static_cast<int64_t>(p.size(0)) });

        solve_for_dp();

        if (rho()) {
            update_p();
            lambda_lm = std::max(lambda_lm / lm_down, 1e-7);
        }
        else {
            lambda_lm = std::min(lambda_lm * lm_up, 1e7);
        }

        if (iter_n % (2 * p.size(0)) == 0) {
            broyden_jacobian_update();
        }
        else {
            p.set_requires_grad(true);
            torch_jacobian_update();
        }

        iter_n += 1;
    }

private:
    torch::Tensor y_data;
    torch::Tensor x_data;
    torch::Tensor p;
    torch::Tensor dp;
    torch::Tensor J;
    torch::Tensor W;
    torch::Tensor JTW;
    torch::Tensor JTWJ;
    double lambda_lm;
    double eps1;
    double eps2;
    double eps3;
    double eps4;
    double lm_up;
    double lm_down;
    int iter_n = 0;
};

//Usage:

int main() {
    torch::Tensor true_p = torch::tensor({ 20.0, 10.0, 1.0, 50.0 });
    torch::Tensor x_true = torch::linspace(0, 100, 25);
    torch::Tensor y_true = Function::f(x_true, true_p);
    torch::Tensor init_p = true_p + torch::randn({ 4 }) * 2 * 2 + 4;

    LevenbergMarquad lm(y_true, x_true, init_p, 4.0, 10, 1e-3, 1e-3, 1e-3, 1e-3, 11, 9);

    for (int i = 0; i < 10; ++i) {
        lm.step();
    }

    return 0;
}
