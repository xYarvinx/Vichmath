package com.yarvin;

import org.apache.commons.math3.linear.*;
import java.util.Arrays;

public class RelaxMethod {

    public static void main(String[] args) throws Exception {
        // Определение первой системы: матрицы A и вектора b
        double[][] A = {
                {10.9, 1.2, 2.1, 0.9},
                {1.2, 11.2, 1.5, 2.5},
                {2.1, 1.5, 9.8, 1.3},
                {0.9, 2.5, 1.3, 12.1}
        };
        double[] b = {-7.0, 5.3, 10.3, 24.6};


            // Решение первым методом
            double[] solution1 = simpleIteration(A, b, 1e-15, 100);
            System.out.println("\nРешение первым методом: " + Arrays.toString(solution1));

            // Решение прямым методом (LU-разложение)
            double[] x_1 = solveDirect(A, b);

            // Вычисление погрешности между решениями
            double error1 = vectorNorm(subtractVectors(x_1, solution1));
            System.out.println("Погрешность между двумя ответами для первого метода: " + error1);

            // Решение методом релаксации
            double[] solution2 = relaxation(A, b, 1.2, 1e-15, 300000);
            System.out.println("\nРешение вторым методом с омега 1.2: " + Arrays.toString(solution2));

            // Вычисление погрешности между решениями
            double error2 = vectorNorm(subtractVectors(x_1, solution2));
            System.out.println("Погрешность между двумя ответами для второго метода: " + error2);

            double[] solution3 = relaxation(A, b, 1.7, 1e-10, 3000000);
            System.out.println("\nРешение вторым методом с омега 1.3: " + Arrays.toString(solution3));
            System.out.println("Погрешность между двумя ответами для второго метода: " + vectorNorm(subtractVectors(x_1, solution3)));

            double[] solution4 = relaxation(A, b, 1.0, 1e-15, 300000);
            System.out.println("\nРешение вторым методом с омега 1.4: " + Arrays.toString(solution4));
            System.out.println("Погрешность между двумя ответами для второго метода: " + vectorNorm(subtractVectors(x_1, solution4)));


            double[] solution5 = relaxation(A, b, 1.5, 1e-15, 300000);
            System.out.println("\nРешение вторым методом с омега 1.5: " + Arrays.toString(solution5));
            System.out.println("Погрешность между двумя ответами для второго метода: " + vectorNorm(subtractVectors(x_1, solution5)));

            double[] solution6 = relaxation(A, b, 0.5, 1e-15, 300000);
            System.out.println("\nРешение вторым методом с омега 0.5: " + Arrays.toString(solution6));
            System.out.println("Погрешность между двумя ответами для второго метода: " + vectorNorm(subtractVectors(x_1, solution6)));



            // Вычисление погрешноси между методами
            double error3 = vectorNorm(subtractVectors(solution1, solution2));
            System.out.println("Погрешность между двумя ответами методов: " + error3);


    }

    // Проверка матрицы на диагональное преобладание
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
            // Условие диагонального преобладания
            if (diag <= sum) {
                return false;
            }
        }
        return true;
    }

    // Метод простой итерации для решения системы уравнений(Якоби)
    public static double[] simpleIteration(double[][] A_data, double[] b_data, double tolerance, int maxSteps) throws Exception {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        int n = A.getRowDimension();

        // Нормализация матрицы и правой части
        for (int i = 0; i < n; i++) {
            double divisor = A.getEntry(i, i);
            for (int j = 0; j < n; j++) {
                A.setEntry(i, j, A.getEntry(i, j) / divisor);
            }
            b.setEntry(i, b.getEntry(i) / divisor);
        }

        // Проверка диагонального преобладания
        if (!isDiagonallyDominant(A)) {
            throw new Exception("Матрица не является диагонально доминантной. Метод может не сходиться.");
        }

        RealVector x = new ArrayRealVector(n); // Начальное приближение (нулевой вектор)

        // Итерационный процесс
        for (int step = 0; step < maxSteps; step++) {
            RealVector x_new = new ArrayRealVector(n);

            // Вычисление нового значения для каждой переменной
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        sum += A.getEntry(i, j) * x.getEntry(j);
                    }
                }
                x_new.setEntry(i, b.getEntry(i) - sum);
            }

            // Проверка на сходимость
            if (x_new.subtract(x).getNorm() < tolerance) {
                return x_new.toArray();
            }

            x = x_new; // Обновление решения
            System.out.println(Arrays.toString(x.toArray()) + " итерация №: " + (step + 1)); // Вывод текущей итерации
        }

        throw new Exception("Итерации не сошлись за указанное количество шагов.");
    }

    // Метод релаксации для решения системы уравнений
    public static double[] relaxation(double[][] A_data, double[] b_data, double omega, double tolerance, int maxSteps) throws Exception {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        int n = A.getRowDimension();
        RealVector x = new ArrayRealVector(n); // Начальное приближение (нулевой вектор)

        // Итерационный процесс
        for (int step = 0; step < maxSteps; step++) {
            RealVector x_new = x.copy();

            // Обновление каждой переменной с использованием метода релаксации
            for (int i = 0; i < n; i++) {
                double sum1 = 0.0;
                for (int j = 0; j < i; j++) {
                    sum1 += A.getEntry(i, j) * x_new.getEntry(j); // Учет уже обновленных переменных
                }
                double sum2 = 0.0;
                for (int j = i + 1; j < n; j++) {
                    sum2 += A.getEntry(i, j) * x.getEntry(j); // Учет старых значений переменных
                }
                double xi = (1 - omega) * x.getEntry(i) + omega * (b.getEntry(i) - sum1 - sum2) / A.getEntry(i, i);
                x_new.setEntry(i, xi);
            }

            // Проверка на сходимость
            if (x_new.subtract(x).getNorm() < tolerance) {
                return x_new.toArray();
            }

            x = x_new; // Обновление решения
            System.out.println(Arrays.toString(x.toArray()) + " итерация №: " + (step + 1)); // Вывод текущей итерации
        }

        throw new Exception("Итерации не сошлись за указанное количество шагов.");
    }

    // Нахождение нормы вектора
    public static double vectorNorm(double[] v) {
        RealVector vec = new ArrayRealVector(v);
        return vec.getNorm();
    }

    // Вычитание двух векторов
    public static double[] subtractVectors(double[] a, double[] b) {
        RealVector vecA = new ArrayRealVector(a);
        RealVector vecB = new ArrayRealVector(b);
        return vecA.subtract(vecB).toArray();
    }

    // Решение системы уравнений прямым методом (LU-разложение)
    public static double[] solveDirect(double[][] A_data, double[] b_data) {
        RealMatrix A = MatrixUtils.createRealMatrix(A_data);
        RealVector b = MatrixUtils.createRealVector(b_data);

        // Использование LU-разложения для решения
        DecompositionSolver solver = new LUDecomposition(A).getSolver();
        RealVector solution = solver.solve(b);
        return solution.toArray();
    }
}
