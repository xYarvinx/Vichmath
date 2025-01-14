package com.yarvin;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;

public class JacobiEigenvalue {

    /**
     * Функция sign, где sign(0) = 1
     */
    public static double sign(double x) {
        if (x > 0) return 1.0;
        if (x < 0) return -1.0;
        return 1.0;
    }

    /**
     * Вычисляет коэффициенты вращения для метода Якоби.
     *
     * @param a_ii Значение A_ii
     * @param a_jj Значение A_jj
     * @param a_ij Значение A_ij
     * @return Массив из двух элементов: [cos(phi), sin(phi)]
     */
    public static double[] calculateRotationCoefficient(double a_ii, double a_jj, double a_ij) {
        if (a_ij == 0.0) {
            return new double[]{1.0, 0.0};
        }

        double tau = (a_jj - a_ii) / (2.0 * a_ij);
        double t;
        if (tau >= 0) {
            t = 1.0 / (tau + Math.sqrt(1.0 + tau * tau));
        } else {
            t = -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        }

        double c = 1.0 / Math.sqrt(1.0 + t * t);
        double s = t * c;
        return new double[]{c, s};
    }

    /**
     * Метод вращений Якоби для нахождения собственных значений и векторов.
     *
     * @param A             Исходная симметричная матрица
     * @param sigma         Порог сходимости
     * @param maxIterations Максимальное количество итераций
     * @return Массив из двух матриц: [A_working, V]
     */
    public static RealMatrix[] jacobiRotation(RealMatrix A, double sigma, int maxIterations) {
        int n = A.getRowDimension();
        RealMatrix A_working = A.copy();
        RealMatrix V = MatrixUtils.createRealIdentityMatrix(n); // Матрица собственных векторов
        int iterationCount = 0; // Счетчик итераций
        while (iterationCount < maxIterations) {
            // Находим максимальный по модулю внедиагональный элемент
            double maxElement = 0.0;
            int i_max = 0;
            int j_max = 1;

            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    double abs = Math.abs(A_working.getEntry(i, j));
                    if (abs > maxElement) {
                        maxElement = abs;
                        i_max = i;
                        j_max = j;
                    }
                }
            }

            // Проверяем условие сходимости
            if (maxElement < sigma) {
                break;
            }


            iterationCount++; // Увеличиваем счетчиthresholdк итераций

            // Вычисляем коэффициенты вращения
            double a_ii = A_working.getEntry(i_max, i_max);
            double a_jj = A_working.getEntry(j_max, j_max);
            double a_ij = A_working.getEntry(i_max, j_max);
            double[] cs = calculateRotationCoefficient(a_ii, a_jj, a_ij);
            double c = cs[0];
            double s = cs[1];

            // Обновляем элементы матрицы A
            for (int k = 0; k < n; k++) {
                if (k != i_max && k != j_max) {
                    double a_ki = A_working.getEntry(k, i_max);
                    double a_kj = A_working.getEntry(k, j_max);
                    double new_a_ki = c * a_ki - s * a_kj;
                    double new_a_kj = s * a_ki + c * a_kj;
                    A_working.setEntry(k, i_max, new_a_ki);
                    A_working.setEntry(i_max, k, new_a_ki);
                    A_working.setEntry(k, j_max, new_a_kj);
                    A_working.setEntry(j_max, k, new_a_kj);
                }
            }

            double new_a_ii = c * c * a_ii + s * s * a_jj - 2.0 * c * s * a_ij;
            double new_a_jj = s * s * a_ii + c * c * a_jj + 2.0 * c * s * a_ij;

            A_working.setEntry(i_max, i_max, new_a_ii);
            A_working.setEntry(j_max, j_max, new_a_jj);
            A_working.setEntry(i_max, j_max, 0.0);
            A_working.setEntry(j_max, i_max, 0.0);

            // Обновляем собственные векторы
            for (int k = 0; k < n; k++) {
                double v_ki = V.getEntry(k, i_max);
                double v_kj = V.getEntry(k, j_max);
                double new_v_ki = c * v_ki - s * v_kj;
                double new_v_jj = s * v_ki + c * v_kj;
                V.setEntry(k, i_max, new_v_ki);
                V.setEntry(k, j_max, new_v_jj);
            }

            // Вывод количества итераций и текущего максимального элемента
            System.out.printf("Итерация %d: максимальный элемент = %.6f%n", iterationCount, maxElement);
        }

        System.out.println("Общее количество итераций: " + iterationCount);
        return new RealMatrix[]{A_working, V};
    }
    public static void main(String[] args) {
        // Исходная симметричная матрица
        double[][] data = {
                {-0.168700, 0.353699, 0.008540, 0.733624},
                {0.353699, 0.056519, -0.723182, -0.076440},
                {0.008540, -0.723182, 0.015938, 0.342333},
                {0.733624, -0.076440, 0.342333, -0.045744}
        };
        double[][] data1 = {
                {2.2, 1, 0.5, 2},
                {1, 1.3, 2, 1},
                {0.5, 2, 0.5, 1.6},
                {2, 1, 1.6, 2}
        };
        double[][] data2 = {
                {1.00, 0.42, 0.54, 0.66},
                {0.42, 1.00, 0.32, 0.44},
                {0.54, 0.32, 1.00, 0.22},
                {0.66, 0.44, 0.22, 1.00}
        };


        RealMatrix A1 = MatrixUtils.createRealMatrix(data);
        RealMatrix A2 = MatrixUtils.createRealMatrix(data1);
        RealMatrix A3 = MatrixUtils.createRealMatrix(data2);

        // Ожидаемые собственные значения
        double[] expectedEigenvalues = {-0.943568, -0.744036, 0.687843, 0.857774};
        double[] expectedEigenvalues1 = {5.652, 1.5455, -1.420, 0.22266};
        double[] expectedEigenvalues2 = {2.3227, 0.7967, 0.6383, 0.2423};
        Arrays.sort(expectedEigenvalues);
        Arrays.sort(expectedEigenvalues1);
        Arrays.sort(expectedEigenvalues2);

        // Параметры метода Якоби
        double sigma = 1e-10; // Порог сходимости
        int maxIterations = 1000; // Максимальное количество итераций

        // Запуск метода вращения Якоби
        RealMatrix[] result = jacobiRotation(A1, sigma, maxIterations);
        RealMatrix A_working = result[0];
        RealMatrix eigenvectors = result[1];

        // Получаем собственные значения (диагональные элементы результирующей матрицы)
        double[] computedEigenvalues = new double[A_working.getRowDimension()];
        for (int i = 0; i < A_working.getRowDimension(); i++) {
            computedEigenvalues[i] = A_working.getEntry(i, i);
        }
        Arrays.sort(computedEigenvalues);
        double[] sortedExpected = expectedEigenvalues.clone();
        double[] sortedComputed = computedEigenvalues.clone();

        // Вывод результатов
        System.out.println("\nВычисленные собственные значения:");
        System.out.println(Arrays.toString(sortedComputed));

        System.out.println("\nОжидаемые собственные значения:");
        System.out.println(Arrays.toString(sortedExpected));

        // Разница между вычисленными и ожидаемыми значениями
        double[] differences = new double[sortedComputed.length];
        for (int i = 0; i < differences.length; i++) {
            differences[i] = Math.abs(sortedComputed[i] - sortedExpected[i]);
        }

        System.out.println("\nРазница между вычисленными и ожидаемыми собственными значениями:");
        System.out.println(Arrays.toString(differences));

        // 2 матрица

        // Запуск метода вращения Якоби
        RealMatrix[] result1 = jacobiRotation(A2, sigma, maxIterations);
        RealMatrix A_working1 = result1[0];
        RealMatrix eigenvectors1 = result1[1];

        // Получаем собственные значения (диагональные элементы результирующей матрицы)
        double[] computedEigenvalues1 = new double[A_working1.getRowDimension()];
        for (int i = 0; i < A_working1.getRowDimension(); i++) {
            computedEigenvalues1[i] = A_working1.getEntry(i, i);
        }
        Arrays.sort(computedEigenvalues1);
        double[] sortedExpected1 = expectedEigenvalues1.clone();
        double[] sortedComputed1 = computedEigenvalues1.clone();

        // Вывод результатов
        System.out.println("\nВычисленные собственные значения:");
        System.out.println(Arrays.toString(sortedComputed1));

        System.out.println("\nОжидаемые собственные значения:");
        System.out.println(Arrays.toString(sortedExpected1));

        // Разница между вычисленными и ожидаемыми значениями
        double[] differences1 = new double[sortedComputed1.length];
        for (int i = 0; i < differences1.length; i++) {
            differences1[i] = Math.abs(sortedComputed1[i] - sortedExpected1[i]);
        }

        System.out.println("\nРазница между вычисленными и ожидаемыми собственными значениями:");
        System.out.println(Arrays.toString(differences1));

        //3 матрица

        // Запуск метода вращения Якоби
        RealMatrix[] result2 = jacobiRotation(A3, sigma, maxIterations);
        RealMatrix A_working2 = result2[0];
        RealMatrix eigenvectors2 = result2[1];

        // Получаем собственные значения (диагональные элементы результирующей матрицы)
        double[] computedEigenvalues2 = new double[A_working2.getRowDimension()];
        for (int i = 0; i < A_working2.getRowDimension(); i++) {
            computedEigenvalues2[i] = A_working2.getEntry(i, i);
        }
        Arrays.sort(computedEigenvalues2);
        double[] sortedExpected2 = expectedEigenvalues2.clone();
        double[] sortedComputed2 = computedEigenvalues2.clone();

        // Вывод результатов
        System.out.println("\nВычисленные собственные значения:");
        System.out.println(Arrays.toString(sortedComputed2));

        System.out.println("\nОжидаемые собственные значения:");
        System.out.println(Arrays.toString(sortedExpected2));

        // Разница между вычисленными и ожидаемыми значениями
        double[] differences2 = new double[sortedComputed2.length];
        for (int i = 0; i < differences2.length; i++) {
            differences2[i] = Math.abs(sortedComputed2[i] - sortedExpected2[i]);
        }

        System.out.println("\nРазница между вычисленными и ожидаемыми собственными значениями:");
        System.out.println(Arrays.toString(differences2));
    }
}

