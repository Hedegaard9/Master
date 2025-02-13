#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>

namespace py = pybind11;

// Funktion til at beregne kvadratroden af en matrix
py::array_t<double> sqrtm_cpp(py::array_t<double> input_array) {
    // Konverter input fra NumPy til Armadillo matrix
    auto buf = input_array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input skal være en 2D-matrix");
    }

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    arma::mat X(reinterpret_cast<double*>(buf.ptr), rows, cols, false, true);

    // Beregn kvadratroden af matrixen
    arma::cx_mat X_sqrt = arma::sqrtmat(X);

    // Omdan resultatet til en flad array til Python
    auto result = py::array_t<double>({rows, cols});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    std::memcpy(result_ptr, X_sqrt.memptr(), rows * cols * sizeof(double));

    return result;
}

// Eksporter funktionen til Python
PYBIND11_MODULE(sqrtm_cpp, m) {
    m.def("sqrtm_cpp", &sqrtm_cpp, "Compute matrix square root using Armadillo");
}
