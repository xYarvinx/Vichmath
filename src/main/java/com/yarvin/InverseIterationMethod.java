package com.yarvin;

import java.util.Arrays;
import org.apache.commons.math3.linear.*;

public class InverseIterationMethod {

    // Проверка симметричности матрицы
    public static boolean isSymmetric(double[][] A) {
        int n = A.length;
        for(int i = 0; i < n; i++) {
            for(int j = i+1; j < n; j++) {
                if(Math.abs(A[i][j] - A[j][i]) > 1e-10) {
                    return false;
                }
            }
        }
        return true;
    }

    // Обратный итерационный метод
    public static double[] inverseIteration(double[][] A, double sigma, double[] initial, int maxIter, double tol) {
        int n = A.length;

        // Создаём (A - sigma*I)
        double[][] shifted = new double[n][n];
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                shifted[i][j] = A[i][j];
                if(i == j) {
                    shifted[i][j] -= sigma;
                }
            }
        }

        // Используем LU разложение для решения системы (A - sigma*I)y = x
        RealMatrix shiftedMatrix = new Array2DRowRealMatrix(shifted);
        DecompositionSolver solver;
        try {
            LUDecomposition lu = new LUDecomposition(shiftedMatrix);
            solver = lu.getSolver();
            if (!solver.isNonSingular()) {
                throw new ArithmeticException("Матрица (A - sigma*I) вырожденная.");
            }
        } catch (SingularMatrixException e) {
            throw new ArithmeticException("Матрица (A - sigma*I) вырожденная.");
        }

        // Преобразуем начальный вектор в RealVector
        RealVector x = new ArrayRealVector(initial);
        // Нормализуем начальный вектор
        x = x.mapDivide(x.getNorm());

        for(int iter = 0; iter < maxIter; iter++) {
            // Решаем (A - sigma*I)y = x
            RealVector y = solver.solve(x);

            // Нормализуем y
            y = y.mapDivide(y.getNorm());

            // Вычисляем Rayleigh Quotient для оценки собственного значения
            double eigenvalue = rayleighQuotient(y.toArray(), A);

            // Проверяем сходимость по невязке ||Ay - lambda y||
            double[] Ay = multiply(A, y.toArray());
            double[] lambdaY = multiplyByScalar(y.toArray(), eigenvalue);
            double[] residual = subtract(Ay, lambdaY);
            double resNorm = norm(residual);

            if(resNorm < tol) {
                System.out.println("Сходимость достигнута за " + (iter+1) + " итераций.");
                System.out.println("Найденное собственное значение: " + eigenvalue);
                System.out.println("Норма невязки ||Ay - lambda y||: " + resNorm);
                return y.toArray();
            }

            x = y;
        }
        System.out.println("Достигнуто максимальное количество итераций.");
        return x.toArray();
    }

    // Умножение матрицы на вектор
    public static double[] multiply(double[][] A, double[] x) {
        int n = A.length;
        double[] b = new double[n];
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < x.length; j++) {
                b[i] += A[i][j] * x[j];
            }
        }
        return b;
    }

    // Вычитание двух векторов
    public static double[] subtract(double[] a, double[] b) {
        double[] c = new double[a.length];
        for(int i = 0; i < a.length; i++) {
            c[i] = a[i] - b[i];
        }
        return c;
    }

    // Норма вектора (евклидова)
    public static double norm(double[] x) {
        double sum = 0.0;
        for(double xi : x) {
            sum += xi * xi;
        }
        return Math.sqrt(sum);
    }

    // Умножение вектора на скаляр
    public static double[] multiplyByScalar(double[] x, double scalar) {
        double[] y = new double[x.length];
        for(int i = 0; i < x.length; i++) {
            y[i] = x[i] * scalar;
        }
        return y;
    }

    // Нахождение \(\lambda = x^T A x / x^T x\) (Rayleigh Quotient)
    public static double rayleighQuotient(double[] x, double[][] A) {
        double[] Ax = multiply(A, x);
        double numerator = 0.0;
        double denominator = 0.0;
        for(int i = 0; i < x.length; i++) {
            numerator += x[i] * Ax[i];
            denominator += x[i] * x[i];
        }
        return numerator / denominator;
    }

    public static void main(String[] args) {
        double[][] A1 = {
                {-0.168700, 0.353699, 0.008540, 0.733624},
                {0.353699, 0.056519, -0.723182, -0.076440},
                {0.008540, -0.723182, 0.015938, 0.342333},
                {0.733624, -0.076440, 0.342333, -0.045744}
        };

        double[][] A2 = {
                {1.00, 0.42, 0.54, 0.66},
                {0.42, 1.00, 0.32, 0.44},
                {0.54, 0.32, 1.00, 0.22},
                {0.66, 0.44, 0.22, 1.00}
        };

        double[][] A3 = {
                {2.2, 1, 0.5, 2},
                {1, 1.3, 2, 1},
                {0.5, 2, 0.5, 1.6},
                {2, 1, 1.6, 2}
        };

        double[][][] matrices = {A1, A2, A3};
        String[] names = {"A1", "A2", "A3"};

        for(int idx = 0; idx < matrices.length; idx++) {
            double[][] A = matrices[idx];
            String name = names[idx];
            System.out.println("Матрица " + name + ":");
            for(double[] row : A) {
                System.out.println(Arrays.toString(row));
            }

            // Проверка симметричности
            if(isSymmetric(A)) {
                System.out.println("Матрица симметрична.");
            } else {
                System.out.println("Матрица не симметрична.");
            }

            // Параметры итерации
            double sigma = 0.0; // Сдвиг, можно настроить
            double[] initial = {1, 1, 1, 1}; // Начальный вектор
            int maxIter = 1000;
            double tol = 1e-6;

            try {
                System.out.println("=== Обратный итерационный метод ===");
                double[] eigenvector = inverseIteration(A, sigma, initial, maxIter, tol);

                // Вычисляем собственное значение через Rayleigh Quotient внутри метода
                double eigenvalue = rayleighQuotient(eigenvector, A);
                System.out.println("Найденный собственный вектор: " + Arrays.toString(eigenvector));
                System.out.println("Найденное собственное значение: " + eigenvalue);

                // Проверка невязки ||Ay - lambda y||
                double[] Ay = multiply(A, eigenvector);
                double[] lambdaY = multiplyByScalar(eigenvector, eigenvalue);
                double[] residual = subtract(Ay, lambdaY);
                double resNorm = norm(residual);
                System.out.println("Норма невязки ||Ay - lambda y||: " + resNorm);

                // Вычисление истинных собственных значений с помощью Apache Commons Math
                System.out.println("=== Истинные собственные значения ===");
                RealMatrix matrix = new Array2DRowRealMatrix(A);
                EigenDecomposition eigenDecomposition = new EigenDecomposition(matrix);
                double[] realEigenvalues = eigenDecomposition.getRealEigenvalues();
                Arrays.sort(realEigenvalues);
                System.out.println("Истинные собственные значения: " + Arrays.toString(realEigenvalues));

                // Сравнение найденного собственного значения с ближайшим истинным
                double closestEigenvalue = findClosestEigenvalue(eigenvalue, realEigenvalues);
                System.out.println("Ближайшее истинное собственное значение: " + closestEigenvalue);
                System.out.println("Разница: " + Math.abs(eigenvalue - closestEigenvalue));

            } catch (ArithmeticException e) {
                System.out.println("Ошибка: " + e.getMessage());
            }

            System.out.println("----------------------------------------------------\n");
        }
    }

    // Нахождение ближайшего истинного собственного значения к найденному
    public static double findClosestEigenvalue(double eigenvalue, double[] trueEigenvalues) {
        double closest = trueEigenvalues[0];
        double minDiff = Math.abs(eigenvalue - closest);
        for(int i = 1; i < trueEigenvalues.length; i++) {
            double diff = Math.abs(eigenvalue - trueEigenvalues[i]);
            if(diff < minDiff) {
                minDiff = diff;
                closest = trueEigenvalues[i];
            }
        }
        return closest;
    }
}
