package com.yarvin;

import org.apache.commons.math3.linear.*;
import java.util.Arrays;

public class RelaxMethod {

    public static void main(String[] args) {
        // Define matrices A and b
        double[][] A = {
                {10.9, 1.2, 2.1, 0.9},
                {1.2, 11.2, 1.5, 2.5},
                {2.1, 1.5, 9.8, 1.3},
                {0.9, 2.5, 1.3, 12.1}
        };
        double[] b = {-7.0, 5.3, 10.3, 24.6};

        double[][] A1 = {
                {3.82, 1.02, 0.75, 0.81},
                {1.05, 4.53, 0.98, 1.53},
                {0.73, 0.85, 4.71, 0.81},
                {0.88, 0.81, 1.28, 3.50}
        };
        double[] b1 = {15.655, 22.705, 23.480, 16.110};

        // First system
        try {
            double[] solution1 = simpleIteration(A, b, 1e-15, 10000);
            System.out.println("\nРешение для первой системы: " + Arrays.toString(solution1));

            // Solve using direct method
            double[] x_1 = solveDirect(A, b);

            double error1 = vectorNorm(subtractVectors(x_1, solution1));
            System.out.println("Погрешность между двумя ответами для первой матрицы: " + error1);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        // Second system
        try {
            double[] solution2 = relaxation(A1, b1, 1.25, 1e-15, 10000);
            System.out.println("\nРешение для второй системы: " + Arrays.toString(solution2));

            // Solve using direct method
            double[] x_2 = solveDirect(A1, b1);

            double error2 = vectorNorm(subtractVectors(x_2, solution2));
            System.out.println("Погрешность между двумя ответами для второй матрицы: " + error2);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public static boolean isDiagonallyDominant(RealMatrix matrix) {
        int n = matrix.getRowDimension();
        for (int i = 0; i < n; i++) {
            double diag = Math.abs(matrix.getEntry(i, i));
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += Math.abs(matrix.getEntry(i, j));
                }
            }
            if (diag <= sum) {
                return false;
            }
        }
        return true;
    }

    public static double[] simpleIteration(double[][] A_data, double[] b_data, double tolerance, int maxSteps) throws Exception {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        int n = A.getRowDimension();

        // Normalize matrix
        for (int i = 0; i < n; i++) {
            double divisor = A.getEntry(i, i);
            for (int j = 0; j < n; j++) {
                A.setEntry(i, j, A.getEntry(i, j) / divisor);
            }
            b.setEntry(i, b.getEntry(i) / divisor);
        }

        if (!isDiagonallyDominant(A)) {
            throw new Exception("Матрица не является диагонально доминантной. Метод может не сходиться.");
        }

        RealVector x = new ArrayRealVector(n);

        for (int step = 0; step < maxSteps; step++) {
            RealVector x_new = new ArrayRealVector(n);

            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        sum += A.getEntry(i, j) * x.getEntry(j);
                    }
                }
                x_new.setEntry(i, b.getEntry(i) - sum);
            }

            if (x_new.subtract(x).getNorm() < tolerance) {
                return x_new.toArray();
            }

            x = x_new;
            System.out.println(Arrays.toString(x.toArray()));
        }

        throw new Exception("Итерации не сошлись за указанное количество шагов.");
    }

    public static double[] relaxation(double[][] A_data, double[] b_data, double omega, double tolerance, int maxSteps) throws Exception {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        int n = A.getRowDimension();
        RealVector x = new ArrayRealVector(n);

        for (int step = 0; step < maxSteps; step++) {
            RealVector x_new = x.copy();

            for (int i = 0; i < n; i++) {
                double sum1 = 0.0;
                for (int j = 0; j < i; j++) {
                    sum1 += A.getEntry(i, j) * x_new.getEntry(j);
                }
                double sum2 = 0.0;
                for (int j = i + 1; j < n; j++) {
                    sum2 += A.getEntry(i, j) * x.getEntry(j);
                }
                double xi = (1 - omega) * x.getEntry(i) + omega * (b.getEntry(i) - sum1 - sum2) / A.getEntry(i, i);
                x_new.setEntry(i, xi);
            }

            if (x_new.subtract(x).getNorm() < tolerance) {
                return x_new.toArray();
            }

            x = x_new;
            System.out.println(Arrays.toString(x.toArray()));
        }

        throw new Exception("Итерации не сошлись за указанное количество шагов.");
    }

    public static double vectorNorm(double[] v) {
        RealVector vec = new ArrayRealVector(v);
        return vec.getNorm();
    }

    public static double[] subtractVectors(double[] a, double[] b) {
        RealVector vecA = new ArrayRealVector(a);
        RealVector vecB = new ArrayRealVector(b);
        return vecA.subtract(vecB).toArray();
    }

    public static double[] solveDirect(double[][] A_data, double[] b_data) {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        DecompositionSolver solver = new LUDecomposition(A).getSolver();
        RealVector solution = solver.solve(b);
        return solution.toArray();
    }
}

