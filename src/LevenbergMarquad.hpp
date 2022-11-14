#ifndef SABR_TORCH_LM_H
#define SABR_TORCH_LM_H

#include <torch/torch.h>

template <typename T>
int length(T n) {
    int l = 0;
    do {
        l++;
        n /= 10;
    } while(n);
    return l;
}

class LevenbergMarquad {
public:
    LevenbergMarquad(torch::Tensor init_p,torch::Tensor x, torch::Tensor y, torch::Tensor func) {

          unsigned int iter_n = 0;

        // https://pytorch.org/cppdocs/api/function_namespaceat_1a095f3dd9bd82e1754ad607466e32d8e2.html?highlight=detach#_CPPv4N2at11detach_copyERKN2at6TensorE
        // https://pytorch.org/cppdocs/api/classat_1_1_tensor.html?highlight=requires_grad_#_CPPv4NK2at6Tensor14requires_grad_Eb

        p = torch::detach_copy(init_p);
        torch::requires_grad(true);
        {
            torch::NoGradGuard no_grad;
            torch::Tensor J = torch::autograd::grad(func, p);
            const torch::autograd::variable_list J = torch::autograd::grad(const std::vector<at::Tensor> &func, p);
        }
        torch::Tensor W = torch::diag(torch::Tensor((1 / pow(sigma, 2)) * length(x)));
    }

// https://pytorch.org/cppdocs/api/typedef_namespacetorch_1abf2c764801b507b6a105664a2406a410.html?highlight=torch%20no_grad
    void broyden_jacobian_update(torch::Tensor J,torch::Tensor dp) {
        torch::NoGradGuard no_grad;

        torch::Tensor df = func(p + dp) - func(p);
        J += torch::outer(df - torch::mv(J, dp),dp)::div(torch::linalg::norm(dp, ord = 2));
    }

    void torch_jacobian_update(torch::Tensor &p) {
        //         Finite-difference Jacobian update
//        torch::NoGradGuard no_grad;
//        J = torch::autograd::functional::jacobian(func, p);
        // TODO
        return;

    }

    void solve_for_dp() {
        // Solver for optimizer step
        torch::NoGradGuard no_grad;
        static torch::Tensor JTW = torch::matmul(torch::transpose(J, 0, 1), this->W);
        static torch::Tensor JTWJ = torch::matmul(JTW, J);

        torch::Tensor dy = y - func(p);
        torch::Tensor dp = torch::linalg::solve(JTWJ + lambda_lm * torch::diag(torch::diagonal(JTWJ)), torch::mv(JTW, dy));
    }


    void chi_2(torch::Tensor p,torch::Tensor& W,torch::Tensor& y) {
        torch::NoGradGuard no_grad;
//
//        chi2 = y^T.W.y + 2 * y^T.W . y_hat +  (y-hat)^T.W.y_hat
//
        auto y_hat = func(p);

        return (torch::dot(y, torch::mv(W, y)) - 2
        * torch::dot(y, torch::mv(W, y_hat)) +
        torch::dot(y_hat, torch::mv(W, y_hat)));
    }

    bool rho(torch::Tensor W,torch::Tensor y,torch::Tensor dp, torch::Tensor JTWJ) {
        torch::NoGradGuard no_grad;

//    rho =  chi2(p) - chi2(p + dp) / (dp)^T . ( lambda * diag(J^T W J).dp + J^T W . dy )

        dy = y - func(p);
        float i_rho = ((chi_2(p,W,y) - chi_2(p + dp,W,y).div(torch::dot(dp, torch::mv(lambda_lm
                                                                                * torch::diag(torch::diagonal(JTWJ)),
                                                                                dp) + torch::mv(JTW, dy))));

        if (i_rho > this->eps4)
            return true;
        else
            return false;
    }

    void update_p(const torch::Tensor& dp, torch::Tensor p) {
        torch::NoGradGuard no_grad;
        p += dp;
    }

    void step(torch::Tensor J,torch::Tensor dp) {

        dp = 0;

        solve_for_dp();

        if (rho()) {
            update_p(dp);
            lambda_lm = torch::maximum(lambda_lm / lm_down, torch::Tensor(1e-7));
        } else
            lambda_lm = torch::minimum(lambda_lm * lm_up, torch::Tensor(1e7));

        if (iter_n % (2 * len(p)) == 0)
            broyden_jacobian_update(J,dp);
        else {
            p.requires_grad_(true);
            torch_jacobian_update(p);
            iter_n++;
            return p;
        }

        void condition1() {
            //    Add condition for Broyden 1-rank method convergence
            return;
        }
    }

private:
    torch::Tensor x;
    torch::Tensor y;
    torch::Tensor p;
    torch::Tensor func;
    float eps1 = 1e-3;
    float eps2 = 1e-3;
    float eps3 = 1e-3;
    float eps4 = 1e-3;
    float lm_up = 11;
    float lm_down = 9;
    float lambda_lm = 10;
    float sigma = 4.0;
};

#endif //SABR_TORCH_LM_H
