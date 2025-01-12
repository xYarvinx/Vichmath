package com.yarvin;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Locale;

public class SquareRootMethod {

    // Метод для решения системы линейных уравнений с использованием метода квадратного корня
    public static Complex[] squareRootMethod(double[][] A, double[] b) {
        int n = A.length;
        Complex[][] B = new Complex[n][n]; // Матрица B, получаемая в ходе разложения
        Complex[] y = new Complex[n]; // Промежуточный вектор y
        Complex[] x = new Complex[n]; // Вектор решения x

        // Инициализация массивов B, y и x комплексными нулями
        for (int i = 0; i < n; i++) {
            y[i] = Complex.ZERO;
            x[i] = Complex.ZERO;
            for (int j = 0; j < n; j++) {
                B[i][j] = Complex.ZERO;
            }
        }

        // Вычисление элементов матрицы B и вектора y
        for (int i = 0; i < n; i++) {
            Complex sumBDiag = Complex.ZERO;
            // Суммирование квадратов элементов столбца для диагонального элемента
            for (int k = 0; k < i; k++) {
                sumBDiag = sumBDiag.add(B[k][i].multiply(B[k][i]));
            }
            Complex diagonalValue = new Complex(A[i][i], 0.0).subtract(sumBDiag);

            // Проверка на положительность диагонального элемента
            if (diagonalValue.getReal() < 0) {
                diagonalValue = new Complex(diagonalValue.getReal(), 0.0);
            }
            B[i][i] = diagonalValue.sqrt(); // Присваивание корня для диагонального элемента

            // Вычисление элементов выше диагонали
            for (int j = i + 1; j < n; j++) {
                Complex sumBOffDiag = Complex.ZERO;
                for (int k = 0; k < i; k++) {
                    sumBOffDiag = sumBOffDiag.add(B[k][i].multiply(B[k][j]));
                }
                B[i][j] = new Complex(A[i][j], 0.0).subtract(sumBOffDiag).divide(B[i][i]);
            }

            // Вычисление элементов вектора y
            Complex sumY = Complex.ZERO;
            for (int k = 0; k < i; k++) {
                sumY = sumY.add(B[k][i].multiply(y[k]));
            }
            y[i] = new Complex(b[i], 0.0).subtract(sumY).divide(B[i][i]);
        }

        // Вычисление элементов вектора x с помощью обратной подстановки
        for (int i = n - 1; i >= 0; i--) {
            Complex sumX = Complex.ZERO;
            for (int k = i + 1; k < n; k++) {
                sumX = sumX.add(B[i][k].multiply(x[k]));
            }
            x[i] = y[i].subtract(sumX).divide(B[i][i]);
        }

        return x; // Возвращаем вектор решения x
    }


    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        // Определение матриц и векторов
        double[][][] matrices = {
                {
                        {0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245},
                        {0.421, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101},
                        {-0.333, 0.139, 0.502, 0.901, 0.243, 0.819, 0.321},
                        {0.313, -0.409, 0.901, 0.865, 0.423, 0.118, 0.183},
                        {-0.141, 0.321, 0.243, 0.423, 1.274, 0.712, 0.423},
                        {-0.381, 0.0625, 0.819, 0.118, 0.712, 0.808, 0.923},
                        {0.245, 0.101, 0.321, 0.183, 0.423, 0.923, 1.105}
                },
                {
                        {0.512, 0.201, -0.423, 0.431, -0.231, 0.182, 0.054},
                        {0.201, 0.603, 0.318, -0.220, 0.432, -0.311, 0.201},
                        {-0.423, 0.318, 0.651, 0.209, 0.515, 0.118, 0.099},
                        {0.431, -0.220, 0.209, 0.723, 0.319, 0.140, -0.272},
                        {-0.231, 0.432, 0.515, 0.319, 0.812, -0.205, 0.415},
                        {0.182, -0.311, 0.118, 0.140, -0.205, 0.913, -0.317},
                        {0.054, 0.201, 0.099, -0.272, 0.415, -0.317, 0.645}
                },
                {
                        {0.421, 0.314, -0.352, 0.211, -0.451, 0.382, -0.105},
                        {0.314, 0.805, 0.129, -0.120, 0.232, 0.181, 0.061},
                        {-0.352, 0.129, 0.702, 0.309, -0.119, 0.142, 0.211},
                        {0.211, -0.120, 0.309, 0.803, 0.419, -0.110, -0.122},
                        {-0.451, 0.232, -0.119, 0.419, 0.612, 0.205, -0.115},
                        {0.382, 0.181, 0.142, -0.110, 0.205, 0.733, -0.117},
                        {-0.105, 0.061, 0.211, -0.122, -0.115, -0.117, 0.845}
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

        // Цикл по матрицам и векторам
        for (int i = 0; i < matrices.length; i++) {
            System.out.printf("Матрица A_%d и вектор f_%d:\n", i + 1, i + 1);

            double[][] A = matrices[i];
            double[] b = vectors[i];

            // Решение системы методом квадратного корня
            Complex[] x = squareRootMethod(A, b);

            // Вывод решения
            System.out.print("Решение: [");
            for (int j = 0; j < x.length; j++) {
                System.out.printf(" %.8f+%.1fj", x[j].getReal(), x[j].getImaginary());
                if (j < x.length - 1) {
                    System.out.print(",");
                }
            }
            System.out.println(" ]");

            // Проверка с помощью Apache Commons Math
            RealMatrix matrixA = new Array2DRowRealMatrix(A);
            RealVector vectorB = new ArrayRealVector(b);
            DecompositionSolver solver = new LUDecomposition(matrixA).getSolver();
            RealVector result = solver.solve(vectorB);

            // Преобразование решения в действительную часть для сравнения
            double[] realPartOfX = new double[x.length];
            for (int j = 0; j < x.length; j++) {
                realPartOfX[j] = x[j].getReal();
            }

            // Вычисление норм
            RealVector realVectorX = new ArrayRealVector(realPartOfX);
            double normDiff = result.subtract(realVectorX).getNorm();
            double normAxMinusB = matrixA.operate(result).subtract(vectorB).getNorm();

            // Вывод норм
            System.out.printf("Норма разности между решениями: %.15e\n", normDiff);
            System.out.printf("Норма разности (Ax - b): %.15e\n", normAxMinusB);

            System.out.println();
        }
    }
}

