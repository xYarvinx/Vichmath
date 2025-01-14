package com.yarvin;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

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
     * Метод вращений с барьерами для нахождения собственных значений.
     *
     * @param A Исходная симметричная матрица
     * @param p Количество барьеров (уровней sigma)
     * @param maxIterations Максимальное количество итераций для предотвращения бесконечного цикла
     * @return Массив из двух матриц: [D, V]
     */
    public static RealMatrix[] rotationWithBarriers(RealMatrix A, int p, int maxIterations) {
        RealMatrix D = A.copy();
        int n = D.getRowDimension();
        RealMatrix V = MatrixUtils.createRealIdentityMatrix(n); // Матрица собственных векторов
        int counter = 0;

        for (int K = 1; K <= p; K++) {
            double sigma = computeSigma(D, K);
            System.out.printf("K: %d, sigma: %.10f%n", K, sigma);

            while (true) {
                if (counter > 1e5) {
                    throw new IllegalArgumentException("Infinite cycle detected.");
                }

                // Поиск максимального по модулю внедиагонального элемента, превышающего sigma
                double maxVal = -Double.MAX_VALUE;
                int i_max = -1;
                int j_max = -1;

                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            double val = Math.abs(D.getEntry(i, j));
                            if (val > maxVal && val >= sigma) {
                                maxVal = val;
                                i_max = i;
                                j_max = j;
                            }
                        }
                    }
                }

                if (maxVal < sigma || i_max == -1 || j_max == -1) {
                    break; // Нет элементов, превышающих sigma
                }

                // Выполнение вращения
                double a_ii = D.getEntry(i_max, i_max);
                double a_jj = D.getEntry(j_max, j_max);
                double a_ij = D.getEntry(i_max, j_max);

                double d = Math.sqrt(Math.pow(a_ii - a_jj, 2) + 4 * Math.pow(a_ij, 2));
                double s = sign(a_ij * (a_ii - a_jj)) * Math.sqrt(0.5 * (1 - Math.abs(a_ii - a_jj) / d));
                double c = Math.sqrt(0.5 * (1 + Math.abs(a_ii - a_jj) / d));

                // Создание копии для обновления
                RealMatrix C = D.copy();

                for (int k = 0; k < n; k++) {
                    if (k != i_max && k != j_max) {
                        double D_ki = D.getEntry(k, i_max);
                        double D_kj = D.getEntry(k, j_max);
                        C.setEntry(k, i_max, c * D_ki + s * D_kj);
                        C.setEntry(i_max, k, c * D_ki + s * D_kj);
                        C.setEntry(k, j_max, -s * D_ki + c * D_kj);
                        C.setEntry(j_max, k, -s * D_ki + c * D_kj);
                    }
                }

                // Обновление диагональных элементов
                C.setEntry(i_max, i_max, c * c * a_ii + 2 * c * s * a_ij + s * s * a_jj);
                C.setEntry(j_max, j_max, s * s * a_ii - 2 * c * s * a_ij + c * c * a_jj);

                // Обнуление внедиагонального элемента
                C.setEntry(i_max, j_max, 0.0);
                C.setEntry(j_max, i_max, 0.0);

                // Обновление матрицы D
                D = C;

                // Обновление собственных векторов
                for (int k = 0; k < n; k++) {
                    double v_ki = V.getEntry(k, i_max);
                    double v_kj = V.getEntry(k, j_max);
                    double new_v_ki = c * v_ki + s * v_kj;
                    double new_v_kj = -s * v_ki + c * v_kj;
                    V.setEntry(k, i_max, new_v_ki);
                    V.setEntry(k, j_max, new_v_kj);
                }

                counter++;
            }
        }

        System.out.println("Общее количество шагов: " + counter);
        return new RealMatrix[]{D, V};
    }

    /**
     * Вычисляет sigma по формуле:
     * sigma = sqrt(max(abs(diag(D)))) * 10^(-K)
     *
     * @param D Матрица, из диагональных элементов которой будет вычислено sigma
     * @param K Степень десяти
     * @return Вычисленное значение sigma
     */
    public static double computeSigma(RealMatrix D, int K) {
        double maxAbsDiagonal = 0.0;
        for (int i = 0; i < D.getRowDimension(); i++) {
            double abs = Math.abs(D.getEntry(i, i));
            if (abs > maxAbsDiagonal) {
                maxAbsDiagonal = abs;
            }
        }
        return Math.sqrt(maxAbsDiagonal) * Math.pow(10, -K);
    }

    /**
     * Извлекает собственные значения из диагональных элементов матрицы и сортирует их.
     *
     * @param A_working Матрица после метода вращений
     * @return Отсортированный массив собственных значений
     */
    public static double[] extractEigenvalues(RealMatrix A_working) {
        double[] eigenvalues = new double[A_working.getRowDimension()];
        for (int i = 0; i < A_working.getRowDimension(); i++) {
            eigenvalues[i] = A_working.getEntry(i, i);
        }
        Arrays.sort(eigenvalues);
        return eigenvalues;
    }

    /**
     * Сравнивает вычисленные собственные значения с ожидаемыми и выводит разницу.
     *
     * @param computed   Вычисленные собственные значения
     * @param expected   Ожидаемые собственные значения
     */
    public static void compareEigenvalues(double[] computed, double[] expected) {
        System.out.println("\nВычисленные собственные значения:");
        System.out.println(Arrays.toString(computed));

        System.out.println("Ожидаемые собственные значения:");
        System.out.println(Arrays.toString(expected));

        // Разница между вычисленными и ожидаемыми значениями
        double[] differences = new double[computed.length];
        for (int i = 0; i < differences.length; i++) {
            differences[i] = Math.abs(computed[i] - expected[i]);
        }

        System.out.println("Разница между вычисленными и ожидаемыми собственными значениями:");
        System.out.println(Arrays.toString(differences));
    }

    public static void main(String[] args) {
        // Исходные симметричные матрицы
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

        // Параметры метода вращений с барьерами
        int p = 4; // Количество барьеров
        int maxIterations = 100000; // Максимальное количество итераций

        // Запуск метода вращений с барьерами для первой матрицы A1
        System.out.println("=== Обработка матрицы A1 ===");
        RealMatrix[] result1 = rotationWithBarriers(A1, p, maxIterations);
        RealMatrix D1 = result1[0];
        RealMatrix eigenvectors1 = result1[1];
        double[] computedEigenvalues1Computed = extractEigenvalues(D1);
        compareEigenvalues(computedEigenvalues1Computed, expectedEigenvalues);

        // Запуск метода вращений с барьерами для второй матрицы A2
        System.out.println("\n=== Обработка матрицы A2 ===");
        RealMatrix[] result2 = rotationWithBarriers(A2, p, maxIterations);
        RealMatrix D2 = result2[0];
        RealMatrix eigenvectors2 = result2[1];
        double[] computedEigenvalues2Computed = extractEigenvalues(D2);
        compareEigenvalues(computedEigenvalues2Computed, expectedEigenvalues1);

        // Запуск метода вращений с барьерами для третьей матрицы A3
        System.out.println("\n=== Обработка матрицы A3 ===");
        RealMatrix[] result3 = rotationWithBarriers(A3, p, maxIterations);
        RealMatrix D3 = result3[0];
        RealMatrix eigenvectors3 = result3[1];
        double[] computedEigenvalues3Computed = extractEigenvalues(D3);
        compareEigenvalues(computedEigenvalues3Computed, expectedEigenvalues2);
    }
}
