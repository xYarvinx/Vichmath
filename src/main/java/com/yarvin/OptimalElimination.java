package com.yarvin;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;

public class OptimalElimination {
    public static void main(String[] args) {
        // Матрицы и векторы
        double[][][] matrices = {
                {
                        {0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245},
                        {0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101},
                        {0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321},
                        {0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183},
                        {0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423},
                        {0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923},
                        {0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105}
                },
                {
                        {0.512, 0.201, -0.423, 0.431, -0.231, 0.182, 0.054},
                        {0.104, 0.603, 0.318, -0.220, 0.432, -0.311, 0.201},
                        {-0.322, 0.127, 0.651, 0.209, 0.515, 0.118, 0.099},
                        {0.402, -0.143, 0.215, 0.723, 0.319, 0.140, -0.272},
                        {0.172, 0.051, 0.215, -0.312, 0.812, -0.205, 0.415},
                        {0.209, -0.205, 0.112, 0.199, 0.145, 0.913, -0.317},
                        {0.119, 0.124, -0.245, -0.301, 0.411, 0.222, 0.645}
                },
                {
                        {0.421, 0.314, -0.352, 0.211, -0.451, 0.382, -0.105},
                        {0.114, 0.805, 0.129, -0.120, 0.232, 0.181, 0.061},
                        {0.032, -0.227, 0.702, 0.309, -0.119, 0.142, 0.211},
                        {0.492, 0.123, 0.118, 0.803, 0.419, -0.110, -0.122},
                        {-0.172, 0.251, -0.115, 0.212, 0.612, 0.205, -0.115},
                        {0.102, -0.105, -0.011, 0.289, 0.145, 0.733, -0.117},
                        {0.221, -0.124, 0.115, 0.201, -0.411, 0.122, 0.845}
                },
                {
                        {-7.494, 6.929, 0.919, -1.569, -2.978, 0.838, 1.840},
                        {6.929, -6.128, 4.458, -2.866, -5.117, 1.262, 2.273},
                        {0.919, 4.458, -3.906, -2.103, -4.647, 0.311, -0.875},
                        {-1.569, -2.866, -2.103, -9.897, 2.463, -0.254, 0.085},
                        {-2.978, -5.117, -4.647, 2.463, -8.634, -1.655, -1.792},
                        {0.838, 1.262, 0.311, -0.254, -1.655, -8.413, -0.778},
                        {1.840, 2.273, -0.875, 0.085, -1.792, -0.778, -12.197}
                },
                {
                        {4.0, 1.2, 0.8, 0.5, 0.4, 0.3, 0.2},
                        {1.2, 3.8, 1.1, 0.9, 0.5, 0.4, 0.3},
                        {0.8, 1.1, 4.2, 1.0, 0.7, 0.5, 0.4},
                        {0.5, 0.9, 1.0, 4.5, 1.2, 0.8, 0.6},
                        {0.4, 0.5, 0.7, 1.2, 3.6, 1.0, 0.7},
                        {0.3, 0.4, 0.5, 0.8, 1.0, 3.7, 0.9},
                        {0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 3.9}
                }
        };

        double[][] vectors = {
                {0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673},
                {0.201, 1.132, 1.041, 0.914, 1.213, 1.435, 1.102},
                {0.305, 1.452, 1.021, 1.013, 0.923, 1.237, 1.375},
                {0.293, -0.714, 1.866, 0.474, -1.191, 0.657, -0.975},
                {1.5, 2.2, 3.1, 4.0, 3.7, 2.8, 1.9}
        };

        // Решение систем
        for (int idx = 0; idx < matrices.length; idx++) {
            double[][] A = matrices[idx];
            double[] f = vectors[idx];

            RealMatrix matrix = new Array2DRowRealMatrix(A);
            RealVector vector = new ArrayRealVector(f);

            double[] solution = solveByOptimalElimination(A, f);
            RealVector x = new ArrayRealVector(solution);

            // Решение с использованием LU-разложения
            DecompositionSolver solver = new LUDecomposition(matrix).getSolver();
            RealVector xNumpy = solver.solve(vector);

            // Нормы
            double differenceNorm = x.subtract(xNumpy).getNorm();
            double residualNorm = matrix.operate(x).subtract(vector).getNorm();

            // Результаты
            System.out.println("\nРезультаты для A_" + (idx + 1) + ":");
            System.out.println("Решение системы x (метод оптимального исключения): ");
            printVector(solution);

            System.out.println("\nРешение системы x (LU-разложение): ");
            printVector(xNumpy.toArray());

            System.out.printf("\nНорма разницы между решениями: %.6e\n", differenceNorm);
            System.out.printf("Норма невязки: %.6e\n", residualNorm);
        }
    }

    private static double[] solveByOptimalElimination(double[][] A, double[] f) {
        int n = A.length;
        double[] x = f.clone();
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, matrix[i], 0, n);
        }

        // Прямой ход с выбором ведущего элемента
        for (int k = 0; k < n; k++) {
            // Выбор ведущего элемента
            int maxRow = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(matrix[i][k]) > Math.abs(matrix[maxRow][k])) {
                    maxRow = i;
                }
            }
            // Обмен строк
            double[] tempRow = matrix[k];
            matrix[k] = matrix[maxRow];
            matrix[maxRow] = tempRow;

            double tempValue = x[k];
            x[k] = x[maxRow];
            x[maxRow] = tempValue;

            // Нормализация строки
            double pivot = matrix[k][k];
            if (pivot == 0) throw new IllegalArgumentException("Zero pivot element");
            for (int j = k; j < n; j++) {
                matrix[k][j] /= pivot;
            }
            x[k] /= pivot;

            // Вычитание строки из последующих
            for (int i = k + 1; i < n; i++) {
                double factor = matrix[i][k];
                for (int j = k; j < n; j++) {
                    matrix[i][j] -= factor * matrix[k][j];
                }
                x[i] -= factor * x[k];
            }
        }

        // Обратный ход
        for (int k = n - 1; k >= 0; k--) {
            for (int i = k - 1; i >= 0; i--) {
                double factor = matrix[i][k];
                matrix[i][k] -= factor * matrix[k][k];
                x[i] -= factor * x[k];
            }
        }

        return x;
    }

    private static void printVector(double[] vector) {
        for (double v : vector) {
            System.out.printf("%.6f ", v);
        }
        System.out.println();
    }
}
