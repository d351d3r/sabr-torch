#ifndef SABR_TORCH_LM_H
#define SABR_TORCH_LM_H


class LevenbergMarquad {
public:
    LevenbergMarquad(x,y,func) {

        int iter_n = 0;

        // https://pytorch.org/cppdocs/api/function_namespaceat_1a095f3dd9bd82e1754ad607466e32d8e2.html?highlight=detach#_CPPv4N2at11detach_copyERKN2at6TensorE
        // https://pytorch.org/cppdocs/api/classat_1_1_tensor.html?highlight=requires_grad_#_CPPv4NK2at6Tensor14requires_grad_Eb
        int p = Tensor::detach_copy(p),torch::requires_grad();
        {
            torch::NoGradGuard no_grad;
            double J = torch::autograd::functional::jacobian(func,p);
        }
        double W = torch::diag(torch::tensor([1 / pow(sigma,2)] * len(x)));
        float lambda_lm = torch::tensor(lambda_lm);

        double eps1 = torch::tensor(eps1);
        double eps2 = torch::tensor(eps2);
        double eps3 = torch::tensor(eps3);
        double eps4 = torch.tensor(eps4);
        double lm_up = torch.tensor(lm_up);
        double lm_down = torch.tensor(lm_down);

    }

// https://pytorch.org/cppdocs/api/typedef_namespacetorch_1abf2c764801b507b6a105664a2406a410.html?highlight=torch%20no_grad
    void broyden_jacobian_update() {
        torch::NoGradGuard no_grad;

        df = func(p + dp) - func(p);
        J += torch::outer(df - torch::mv(J, dp),
                         dp)::div(torch::linalg::norm(dp, ord = 2));
    }
    void torch_jacobian_update(int &p) {
//        torch::NoGradGuard no_grad;
//        J = torch::autograd::functional::jacobian(func, p);
        // TODO
        return;

    }

    void solve_for_dp() {
        torch::NoGradGuard no_grad;
        auto JTW = torch::matmul(torch::transpose(J, 0, 1), W);
        auto JTWJ = torch::matmul(JTW, J);

        auto dy = y_data - func(p);
        auto dp = torch::linalg::solve(JTWJ + lambda_lm * torch::diag(torch::diagonal(JTWJ)),torch::mv(JTW, dy));
    }


    void chi_2(p) {
        torch::NoGradGuard no_grad;
//        '''
//
//        chi2 = y^T.W.y + 2 * y^T.W . y_hat +  (y-hat)^T.W.y_hat
//
//        '''
        auto y_hat = func(p);

        return (torch::dot(y_data, torch::mv(W, y_data)) -2
        * torch::dot(y_data, torch::mv(W, y_hat)) + torch::dot(y_hat, torch.mv(W,y_hat)));
    }

    void rho() {
        torch::NoGradGuard no_grad;

//    '''
//    rho =  chi2(p) - chi2(p + dp) / (dp)^T . ( lambda * diag(J^T W J).dp + J^T W . dy )
//
//    '''

        dy = self.y_data - self.func(self.p)
        rho = ((self.chi_2(self.p) - self.chi_2(self.p + self.dp))
                .div(torch.dot(self.dp, torch.mv(self.lambda_lm * torch.diag(torch.diagonal(self.JTWJ)), self.dp)
                                        + torch.mv(self.JTW, dy)
                     )
                )
        )

        if rho > self.eps4:
        return True
        else:
        return False
    }
    void update_p(dp) {
        torch::NoGradGuard no_grad;
        p += dp;
    }


    void step() {

        auto dp = 0;

        solve_for_dp();

        if (rho()) {
            update_p(dp);
            lambda_lm = torch::maximum(lambda_lm / lm_down, torch::tensor(1e-7));
        }
        else
            lambda_lm = torch::minimum(lambda_lm * lm_up, torch::tensor(1e7));

        if (iter_n % (2 * len(p)) == 0)
            broyden_jacobian_update();
        else {

            p.requires_grad_(true);

            torch_jacobian_update(p);

            iter_n += 1;

            return p;
        }
        void condition1() {}
        return;
    }

private:
    int x;
    int y;
    int p;
    int func;
    double eps1 = 1e-3;
    double eps2 = 1e-3;
    double eps3 =  1e-3;
    double eps4 =  1e-3;
    int lm_up = 11;
    int lm_down = 9;
    int lambda_lm = 10;
    double sigma = 4.0;

};


#endif //SABR_TORCH_LM_H
