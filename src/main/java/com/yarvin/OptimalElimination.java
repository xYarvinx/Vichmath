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
                        {1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 5, 0, 0, 0, 0, 0, 0},
                        {0, 0, 3, 0, 0, 0, 0, 0},
                        {0, 0, 0, 3, 0, 0, 0, 0},
                        {0, 0, 0, 0, 7, 0, 0, 0},
                        {0, 0, 0, 0, 0, 6, 0, 0},
                        {0, 0, 0, 0, 0, 0, 8, 0},
                        {0, 0, 0, 0, 0, 0, 0, 4}
                },
                {
                        {9, 5, 6, 6, 3, 8, 1, 2},
                        {5, 9, 8, 3, 7, 6, 3, 7},
                        {6, 8, 7, 8, 5, 7, 2, 4},
                        {6, 3, 8, 7, 3, 4, 2, 8},
                        {3, 7, 5, 3, 3, 5, 6, 6},
                        {8, 6, 7, 4, 5, 9, 4, 8},
                        {1, 3, 2, 2, 6, 4, 4, 7},
                        {2, 7, 4, 8, 6, 8, 7, 7}
                },
                {
                        {6, 1, 8, 5, 1, 3, 7, 4},
                        {0, 1, 1, 4, 7, 7, 1, 7},
                        {0, 0, 9, 2, 5, 5, 4, 7},
                        {0, 0, 0, 4, 5, 2, 3, 3},
                        {0, 0, 0, 0, 6, 1, 8, 2},
                        {0, 0, 0, 0, 0, 8, 7, 6},
                        {0, 0, 0, 0, 0, 0, 5, 1},
                        {0, 0, 0, 0, 0, 0, 0, 3}
                },
                {
                        {7, 5, 3, 7, 2, 9, 2, 5},
                        {2, 5, 7, 9, 1, 5, 3, 5},
                        {2, 3, 7, 4, 9, 9, 7, 8},
                        {7, 9, 5, 5, 6, 4, 8, 8},
                        {1, 9, 6, 4, 7, 1, 5, 8},
                        {6, 8, 8, 8, 6, 6, 5, 2},
                        {4, 6, 9, 3, 7, 8, 4, 7},
                        {9, 4, 5, 7, 6, 7, 3, 8}
                }
        };

        double[][] vectors = {
                {6, 6, 3, 3, 3, 9, 4, 5},
                {1, 1, 8, 6, 6, 5, 2, 5},
                {9, 8, 9, 3, 1, 4, 5, 7},
                {2, 4, 2, 7, 6, 1, 3, 3},
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
