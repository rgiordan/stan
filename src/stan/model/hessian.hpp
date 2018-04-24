#ifndef STAN_MODEL_HESSIAN_HPP
#define STAN_MODEL_HESSIAN_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>

namespace stan {
  namespace model {

    template <class M>
    void hessian(const M& model,
                 const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                 double& f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
                 std::ostream* msgs = 0) {
      stan::math::hessian<model_functional<M> >(model_functional<M>(model,
                                                                    msgs),
                                                x, f, grad_f, hess_f);
    }


    template <bool propto, bool jacobian_adjust_transform, class M>
    void log_prob_hessian(
        const M& model,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
        double& f,
        Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
        std::ostream* msgs = 0) {
      stan::math::hessian<model_functional_template<
        propto, jacobian_adjust_transform, M>>(
          model_functional_template<
            propto, jacobian_adjust_transform, M>(model, msgs),
              x, f, grad_f, hess_f);
    }


    /**
     * Compute the hessian using reverse-mode automatic
     * differentiation, writing the result into the specified
     * gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[out] gradient Vector into which gradient is written.
     * @param[out] hessian matrix into which hessian is written.
     * @param[in,out] msgs
     */
    // template <bool propto, bool jacobian_adjust_transform, class M>
    // double log_prob_hessian(const M& model,
    //                         Eigen::VectorXd& params_r,
    //                         double& f,
    //                         Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
    //                         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
    //                         std::ostream* msgs = 0) {
    //   using std::vector;
    //   using stan::math::var;
    //
    //   Eigen::Matrix<var, Eigen::Dynamic, 1> ad_params_r(params_r.size());
    //   for (size_t i = 0; i < model.num_params_r(); ++i) {
    //     stan::math::var var_i(params_r[i]);
    //     ad_params_r[i] = var_i;
    //   }
    //   try {
    //     var adLogProb
    //       = model
    //       .template log_prob<propto,
    //                          jacobian_adjust_transform>(ad_params_r, msgs);
    //     double val = adLogProb.val();
    //     stan::math::hessian(
    //         adLogProb,
    //         ad_params_r,
    //         grad_f,
    //         hess_f);
    //     return val;
    //   } catch (std::exception &ex) {
    //     stan::math::recover_memory();
    //     throw;
    //   }
    // }

  }

  }
}
#endif
