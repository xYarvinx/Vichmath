package com.yarvin;

import java.util.Arrays;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class SimpleIteration {

    public static class Result {
        public final double[] eigenvector;
        public final double eigenvalue;
        public final int iterations;
        public final double residualNorm;

        public Result(double[] eigenvector, double eigenvalue, int iterations, double residualNorm) {
            this.eigenvector = eigenvector;
            this.eigenvalue = eigenvalue;
            this.iterations = iterations;
            this.residualNorm = residualNorm;
        }
    }

    public static Result simpleIter(
            double[][] A,
            double[] x0,
            double eps,
            int maxIter
    ) {
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

        double eigenvalue = 0;
        for (int iter = 0; iter < maxIter; iter++) {
            double[] y = multiplyMatrixVector(A, x);
            double newEigenvalue = dotProduct(y, x);

            normalize(y);

            if (Math.abs(eigenvalue - newEigenvalue) < eps) {
                double residualNorm = calculateResidualNorm(A, y, newEigenvalue);
                return new Result(y, newEigenvalue, iter, residualNorm);
            }

            x = y;
            eigenvalue = newEigenvalue;
        }

        double residualNorm = calculateResidualNorm(A, x, eigenvalue);
        return new Result(x, eigenvalue, maxIter, residualNorm);
    }

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

    private static void normalize(double[] vector) {
        double norm = Math.sqrt(Arrays.stream(vector).map(v -> v * v).sum());
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= norm;
        }
    }

    private static double dotProduct(double[] vector1, double[] vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }

    private static double calculateResidualNorm(double[][] matrix, double[] eigenvector, double eigenvalue) {
        RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrix);
        RealVector realEigenvector = MatrixUtils.createRealVector(eigenvector);
        RealVector residual = realMatrix.operate(realEigenvector).subtract(realEigenvector.mapMultiply(eigenvalue));
        return residual.getNorm();
    }

    public static void main(String[] args) {
        double[][] matrix1 = {
                {-0.168700, 0.353699, 0.008540, 0.733624},
                {0.353699, 0.056519, -0.723182, -0.076440},
                {0.008540, -0.723182, 0.015938, 0.342333},
                {0.733624, -0.076440, 0.342333, -0.045744}
        };

        double[][] matrix2 = {
                {1.00, 0.42, 0.54, 0.66},
                {0.42, 1.00, 0.32, 0.44},
                {0.54, 0.32, 1.00, 0.22},
                {0.66, 0.44, 0.22, 1.00}
        };

        double[][] matrix3 = {
                {2.2, 1, 0.5, 2},
                {1, 1.3, 2, 1},
                {0.5, 2, 0.5, 1.6},
                {2, 1, 1.6, 2}
        };

        double[][][] matrices = {matrix1, matrix2, matrix3};
        double eps = 1e-16;
        int maxIter = 100000;

        for (double[][] matrix : matrices) {
            System.out.println("Матрица:");
            for (double[] row : matrix) {
                System.out.println(Arrays.toString(row));
            }

            Result result = simpleIter(matrix, null, eps, maxIter);

            System.out.println("Собственный вектор: " + Arrays.toString(result.eigenvector));
            System.out.println("Собственное значение: " + result.eigenvalue);
            System.out.println("Количество итераций: " + result.iterations);
            System.out.println("Норма невязки: " + result.residualNorm);
            System.out.println("====================");
        }
    }
}
