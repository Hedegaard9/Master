#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

// EWMA-funktion
py::array_t<double> ewma_c(py::array_t<double> x, double lambda, int start) {
    auto buf = x.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int n = buf.shape[0];

    std::vector<double> var_vec(n, NAN);
    int na = 0;
    double initial_value = 0;

    if (n <= start) {
        return py::array_t<double>(var_vec.size(), var_vec.data());
    }

    for (int i = 0; i < start; ++i) {
        if (std::isnan(ptr[i])) {
            na += 1;
        } else {
            initial_value += std::pow(ptr[i], 2);
        }
    }
    initial_value /= (start - 1 - na);

    var_vec[start] = initial_value;
    for (int j = start + 1; j < n; ++j) {
        if (std::isnan(ptr[j - 1])) {
            var_vec[j] = var_vec[j - 1];
        } else {
            var_vec[j] = lambda * var_vec[j - 1] + (1 - lambda) * std::pow(ptr[j - 1], 2);
        }
    }

    for (auto &v : var_vec) {
        v = std::sqrt(v);
    }

    return py::array_t<double>(var_vec.size(), var_vec.data());
}

// Eksporter funktion til Python
PYBIND11_MODULE(ewma, m) {
    m.def("ewma_c", &ewma_c, "EWMA Volatility Calculation");
}
