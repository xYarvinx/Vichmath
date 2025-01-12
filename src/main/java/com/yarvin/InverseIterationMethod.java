package com.yarvin;

import org.apache.commons.math3.linear.*;

import java.util.Arrays;

public class InverseIterationMethod {

    public static class Result {
        public final double[] eigenvector;
        public final double eigenvalue;
        public final int iterations;

        public Result(double[] eigenvector, double eigenvalue, int iterations) {
            this.eigenvector = eigenvector;
            this.eigenvalue = eigenvalue;
            this.iterations = iterations;
        }
    }

    // Метод для умножения матрицы на вектор
    private static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int size = matrix.length;
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    // Нормализация вектора
    private static void normalize(double[] vector) {
        double norm = Math.sqrt(Arrays.stream(vector).map(v -> v * v).sum());
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= norm;
        }
    }

    // Скалярное произведение двух векторов
    private static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }

    // Метод для вычисления Rayleigh Quotient
    private static double rayleighQuotient(double[][] A, double[] x) {
        double[] Ax = multiplyMatrixVector(A, x);
        return dotProduct(x, Ax);
    }

    // Метод обратной итерации с фиксированным сдвигом для нахождения наименьшего собственного значения
    public static Result inverseIter(double[][] A, double[] x0, double mu, double eps, int maxIter) {
        int n = A.length;
        double[] x = x0 != null ? x0.clone() : new double[n];

        // Инициализация x случайными значениями, если он не задан
        if (x0 == null) {
            for (int i = 0; i < n; i++) {
                x[i] = Math.random();
            }
        }

        // Нормализация начального вектора
        normalize(x);

        // Создаём (A - μI)
        double[][] shiftedMatrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                shiftedMatrix[i][j] = A[i][j];
                if (i == j) {
                    shiftedMatrix[i][j] -= mu;
                }
            }
        }

        RealMatrix realShiftedMatrix = MatrixUtils.createRealMatrix(shiftedMatrix);
        LUDecomposition luDecomposition;
        try {
            luDecomposition = new LUDecomposition(realShiftedMatrix);
        } catch (SingularMatrixException e) {
            throw new RuntimeException("Смещенная матрица (A - μI) вырождена. Попробуйте другой сдвиг μ.");
        }
        DecompositionSolver solver = luDecomposition.getSolver();

        double eigenvalue = rayleighQuotient(A, x);
        for (int iter = 0; iter < maxIter; iter++) {
            // Преобразуем x в RealVector
            RealVector xVector = new ArrayRealVector(x);

            // Решаем систему (A - μI)y = x
            RealVector yVector;
            try {
                yVector = solver.solve(xVector);
            } catch (SingularMatrixException e) {
                throw new RuntimeException("Система уравнений не имеет решения на итерации " + (iter + 1));
            }

            double[] y = yVector.toArray();

            // Нормализуем новый вектор
            normalize(y);

            // Вычисляем Rayleigh Quotient для обновления собственного значения
            double newEigenvalue = rayleighQuotient(A, y);

            // Проверяем сходимость
            if (Math.abs(newEigenvalue - eigenvalue) < eps) {
                return new Result(y, newEigenvalue, iter + 1);
            }

            // Обновляем вектор и собственное значение
            x = y;
            eigenvalue = newEigenvalue;
        }

        return new Result(x, eigenvalue, maxIter);
    }

    // Метод для вычисления истинного (наименьшего) собственного значения
    private static double getTrueEigenvalue(double[][] A) {
        RealMatrix matrix = MatrixUtils.createRealMatrix(A);
        EigenDecomposition eigenDecomposition = new EigenDecomposition(matrix);
        double[] eigenvalues = eigenDecomposition.getRealEigenvalues();

        // Находим минимальное собственное значение
        double minEigenvalue = Double.POSITIVE_INFINITY;
        for (double eigenvalue : eigenvalues) {
            if (eigenvalue < minEigenvalue) {
                minEigenvalue = eigenvalue;
            }
        }
        return minEigenvalue;
    }

    public static void main(String[] args) {
        double[][] matrix1 = {
                {2.2, 1.0, 0.5, 2.0},
                {1.0, 1.3, 2.0, 1.0},
                {0.5, 2.0, 0.5, 1.6},
                {2.0, 1.0, 1.6, 2.0}
        };

        double[][][] matrices = {matrix1};
        double eps = 1e-10;
        int maxIter = 1000;
        double mu = -2.0; // Фиксированный сдвиг, близкий к наименьшему собственному значению

        for (double[][] matrix : matrices) {
            System.out.println("Матрица:");
            for (double[] row : matrix) {
                System.out.println(Arrays.toString(row));
            }

            try {
                Result result = inverseIter(matrix, null, mu, eps, maxIter);

                System.out.println("Собственный вектор: " + Arrays.toString(result.eigenvector));
                System.out.println("Собственное значение: " + result.eigenvalue);
                System.out.println("Количество итераций: " + result.iterations);

                double trueEigenvalue = getTrueEigenvalue(matrix);
                System.out.println("Вычисленное собственное значение: " + result.eigenvalue);
                System.out.println("Истинное собственное значение: " + trueEigenvalue);
                System.out.println("Невязка: " + Math.abs(trueEigenvalue - result.eigenvalue));
            } catch (RuntimeException e) {
                System.err.println("Ошибка: " + e.getMessage());
            }

            System.out.println("====================");
        }
    }
}
