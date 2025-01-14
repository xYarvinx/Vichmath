package com.yarvin;

import org.apache.commons.math3.linear.*;

/**
 * Класс, реализующий итеративный метод Ричардсона для решения системы линейных уравнений Ax = b.
 *
 * Использует разложение матрицы A на собственные значения для вычисления параметров метода.
 * Оптимизирует шаги итерации с помощью узлов Чебышёва.
 */
public class Richardson {

    private double tol; // Точность
    private double eta; // Отношение собственных значений
    private double tau0; // Оптимальный шаг tau0
    private double rho0; // Коэффициент сходимости rho0
    private RealMatrix A; // Матрица системы A
    private RealVector b; // Правая часть системы b
    private double lambdaMin; // Минимальное положительное собственное значение
    private double lambdaMax; // Максимальное положительное собственное значение
    private int n = 8; // Количество узлов Чебышёва
    private RealVector x; // Текущее приближение решения
    private int p; // Количество знаков точности

    /**
     * Конструктор класса Richardson.
     *
     * @param A Матрица системы A.
     * @param b Вектор правой части b.
     * @param x Начальное приближение решения.
     * @param p Количество знаков точности.
     */
    public Richardson(RealMatrix A, RealVector b, RealVector x, int p) {
        this.p = p;
        this.tol = Math.pow(10, -(p - 1));
        this.A = A;
        this.b = b;
        this.x = x;
    }

    /**
     * Вычисляет спектр матрицы A, определяя минимальное и максимальное положительные собственные значения.
     *
     * @throws Exception Если в спектре матрицы A отсутствуют положительные собственные значения.
     */
    private void computeSpectrum() throws Exception {
        RealMatrix[] eigenResult = JacobiEigenvalue.rotationWithBarriers(A, this.p, 100000);
        double[] eigenvalues = JacobiEigenvalue.extractEigenvalues(eigenResult[0]);

        // Фильтруем положительные собственные значения
        double maxVal = Double.NEGATIVE_INFINITY;
        double minVal = Double.POSITIVE_INFINITY;
        boolean hasPositive = false;

        for (double val : eigenvalues) {
            if (val > 0) {
                hasPositive = true;
                if (val > maxVal) {
                    maxVal = val;
                }
                if (val < minVal) {
                    minVal = val;
                }
            }
        }

        if (!hasPositive) {
            throw new Exception("Нет положительных собственных значений в спектре матрицы A.");
        }

        this.lambdaMax = maxVal;
        this.lambdaMin = minVal;
    }

    /**
     * Вычисляет отношение минимального к максимальному собственному значению (eta).
     */
    private void computeEta() {
        this.eta = this.lambdaMin / this.lambdaMax;
    }

    /**
     * Вычисляет оптимальный шаг tau0.
     */
    private void computeTau0() {
        this.tau0 = 2.0 / (this.lambdaMin + this.lambdaMax);
    }

    /**
     * Вычисляет коэффициент сходимости rho0.
     */
    private void computeRho0() {
        this.rho0 = (1.0 - this.eta) / (1.0 + this.eta);
    }

    /**
     * Вычисляет k-ый узел Чебышёва.
     *
     * @param k Номер узла.
     * @return Значение узла Чебышёва.
     */
    private double v_k(int k) {
        return Math.cos((2.0 * k - 1.0) * Math.PI / (2.0 * this.n));
    }

    /**
     * Вычисляет оптимальный шаг t_k для k-го узла Чебышёва.
     *
     * @param k Номер узла.
     * @return Оптимальный шаг t_k.
     */
    private double t_k(int k) {
        return this.tau0 / (1.0 + this.rho0 * this.v_k(k));
    }

    /**
     * Выполняет одну итерацию метода Ричардсона, обновляя приближение решения.
     *
     * @param xCurrent Текущее приближение решения.
     * @return Новое приближение решения после одной итерации.
     */
    private RealVector computeX(RealVector xCurrent) {
        RealVector xNew = xCurrent.copy();
        for (int k = 1; k <= this.n; k++) {
            double t = this.t_k(k);
            RealVector AxMinusb = A.operate(xNew).mapMultiply(-1.0).add(b);
            xNew = AxMinusb.mapMultiply(t).add(xNew);
        }
        return xNew;
    }

    /**
     * Основной метод, выполняющий итеративный процесс для нахождения решения системы Ax = b.
     *
     * @return Объект класса Result, содержащий решение и количество выполненных итераций.
     * @throws Exception Если метод не сошелся или произошла ошибка при вычислении спектра.
     */
    public Result compute() throws Exception {
        // Шаг 1: Вычисляем спектр матрицы
        computeSpectrum();

        // Шаг 2: Вычисляем отношение собственных значений
        computeEta();

        // Шаг 3: Вычисляем коэффициент сходимости
        computeRho0();

        // Шаг 4: Вычисляем оптимальный шаг tau0
        computeTau0();

        RealVector currentX = this.x.copy();
        int iterations = 0;

        // Итеративный процесс
        while (A.operate(currentX).subtract(b).getNorm() > this.tol) {
            currentX = computeX(currentX);
            iterations += 1;

            if (iterations > 1_000_000) {
                throw new Exception("Метод не сошелся за разумное число итераций.");
            }
        }

        return new Result(currentX, (long) iterations * this.n);
    }

    /**
     * Внутренний класс для хранения результата вычислений.
     */
    public static class Result {
        public RealVector solution; // Вектор решения
        public long totalIterations; // Общее количество итераций

        /**
         * Конструктор класса Result.
         *
         * @param solution Вектор решения.
         * @param totalIterations Общее количество выполненных итераций.
         */
        public Result(RealVector solution, long totalIterations) {
            this.solution = solution;
            this.totalIterations = totalIterations;
        }
    }

    /**
     * Метод для проверки корректности найденного решения.
     *
     * @param A Матрица системы A.
     * @param b Вектор правой части b.
     * @param x Найденное решение x.
     * @param p Количество знаков точности.
     */
    public static void checkAnswer(RealMatrix A, RealVector b, RealVector x, int p) {
        RealVector Ax = A.operate(x);
        RealVector residual = Ax.subtract(b);
        double norm = residual.getNorm();
        double tol = Math.pow(10, -(p - 1));

        if (norm <= tol) {
            System.out.println("Решение корректно в пределах заданной точности.");
        } else {
            System.out.printf("Решение НЕ корректно. Норма остатка: %.10f%n", norm);
        }
    }

    /**
     * Основной метод для тестирования реализации метода Ричардсона.
     *
     * @param args Аргументы командной строки (не используются).
     */
    public static void main(String[] args) {
        try {
            // Пример 1:
            // RealMatrix A = MatrixUtils.createRealMatrix(new double[][]{
            //    {-0.168700, 0.353699, 0.008540, 0.733624},
            //                {0.353699, 0.056519, -0.723182, -0.076440},
            //                {0.008540, -0.723182, 0.015938, 0.342333},
            //                {0.733624, -0.076440, 0.342333, -0.045744}
            // });
            // RealVector b = MatrixUtils.createRealVector(new double[]{-0.943568, -0.744036, 0.687843, 0.857774});
            // RealVector x0 = MatrixUtils.createRealVector(new double[]{0, 0, 0, 0});

            // Пример 2:
            // RealMatrix A = MatrixUtils.createRealMatrix(new double[][]{
            //     {2.2, 1, 0.5, 2},
            //                {1, 1.3, 2, 1},
            //                {0.5, 2, 0.5, 1.6},
            //                {2, 1, 1.6, 2}
            // });
            // RealVector b = MatrixUtils.createRealVector(new double[]{5.652, 1.5455, -1.420, 0.22266});
            // RealVector x0 = MatrixUtils.createRealVector(new double[]{0, 0, 0, 0});

            // Пример 3:
             RealMatrix A = MatrixUtils.createRealMatrix(new double[][]{
                 {1.00, 0.42, 0.54, 0.66},
                            {0.42, 1.00, 0.32, 0.44},
                            {0.54, 0.32, 1.00, 0.22},
                            {0.66, 0.44, 0.22, 1.00}
             });
             RealVector b = MatrixUtils.createRealVector(new double[]{2.3227, 0.7967, 0.6383, 0.2423});
             RealVector x0 = MatrixUtils.createRealVector(new double[]{0, 0, 0, 0});

            // Пример 4:
//            RealMatrix A = MatrixUtils.createRealMatrix(new double[][]{
//                    {2, 1},
//                    {1, 2}
//            });
//            RealVector b = MatrixUtils.createRealVector(new double[]{4, 5});
//            RealVector x0 = MatrixUtils.createRealVector(new double[]{0, 0});

            System.out.println("Собственные значения матрицы A:");
            EigenDecomposition eigenDecomposition = new EigenDecomposition(A);
            double[] eigenvalues = eigenDecomposition.getRealEigenvalues();
            for (double val : eigenvalues) {
                System.out.printf("%.6f ", val);
            }
            System.out.println();



            int precision = 5;

            Richardson richardson = new Richardson(A, b, x0, precision);

            Result result = richardson.compute();

            System.out.println("Вектор решения x:");
            double[] xValues = result.solution.toArray();
            for (double xi : xValues) {
                System.out.printf("%.10f ", xi);
            }
            System.out.println();
            System.out.printf("Итерационный процесс завершился за %d итераций.%n", result.totalIterations);

            // Проверка решения
            checkAnswer(A, b, result.solution, precision);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

