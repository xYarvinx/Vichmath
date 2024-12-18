package com.yarvin;

import org.apache.commons.math3.linear.*;

import java.text.DecimalFormat;

public class Lab4VinnitskiyYaroslav {

    public static void main(String[] args) {
        // Определение матрицы A и вектора b для решения системы A * x = b
        double[][] A = {
                {0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245},
                {0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101},
                {0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321},
                {0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183},
                {0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423},
                {0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923},
                {0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105},
        };
        double[] b = {0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673};

        int n = A.length;

        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        luDecomposition(A, L, U);

        double[] y = forwardSubstitution(L, b);
        double[] x = backwardSubstitution(U, y);

        System.out.println("Решение системы x:");
        printArray(x);

        double[] Ax = multiplyMatrixVector(A, x);
        System.out.println("\nПроверка (A * x):");
        printArray(Ax);
        System.out.println("\nВектор свободных членов b:");
        printArray(b);

        RealMatrix matrixA = new Array2DRowRealMatrix(A);
        RealVector vectorB = new ArrayRealVector(b);
        DecompositionSolver solver = new LUDecomposition(matrixA).getSolver();
        RealVector solutionVector = solver.solve(vectorB);
        double[] solutionFromApache = solutionVector.toArray();

        // Вычисление и вывод первой нормы разности
        double firstNormSolutionDiff = calculateFirstNormDifference(solutionFromApache, x);
        double firstNormAxDiff = calculateFirstNormDifference(Ax, b);

        System.out.println("\nПервая норма разности решения через Apache Commons Math и нашего решения: " + firstNormSolutionDiff);
        System.out.println("Первая норма разности произведения матрицы A на вектор x и вектора b: " + firstNormAxDiff);
    }

    public static void luDecomposition(double[][] A, double[][] L, double[][] U) {
        int n = A.length;

        for (int i = 0; i < n; i++) {
            for (int k = i; k < n; k++) {
                double sumUpper = 0;
                for (int j = 0; j < i; j++) {
                    sumUpper += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sumUpper;
            }

            L[i][i] = 1;

            for (int k = i + 1; k < n; k++) {
                double sumLower = 0;
                for (int j = 0; j < i; j++) {
                    sumLower += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sumLower) / U[i][i];
            }
        }
    }

    public static double[] forwardSubstitution(double[][] L, double[] b) {
        int n = b.length;
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            double sumForward = 0;
            for (int j = 0; j < i; j++) {
                sumForward += L[i][j] * y[j];
            }
            y[i] = b[i] - sumForward;
        }

        return y;
    }

    public static double[] backwardSubstitution(double[][] U, double[] y) {
        int n = y.length;
        double[] x = new double[n];

        for (int i = n - 1; i >= 0; i--) {
            double sumBackward = 0;
            for (int j = i + 1; j < n; j++) {
                sumBackward += U[i][j] * x[j];
            }
            x[i] = (y[i] - sumBackward) / U[i][i];
        }

        return x;
    }

    public static double[] multiplyMatrixVector(double[][] A, double[] x) {
        int n = x.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            result[i] = sum;
        }

        return result;
    }

    public static void printArray(double[] array) {
        DecimalFormat df = new DecimalFormat("0.###############");
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(df.format(array[i]));
            if (i < array.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        System.out.println(sb.toString());
    }

    public static double calculateFirstNormDifference(double[] v1, double[] v2) {
        double norm = 0;
        for (int i = 0; i < v1.length; i++) {
            norm += Math.abs(v1[i] - v2[i]);
        }
        return norm;
    }
}
