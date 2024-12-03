package com.yarvin;

import org.apache.commons.math3.linear.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.*;
import javax.swing.*;
import java.util.ArrayList;
import java.util.List;

public class GradientDescentMethodLab9 {

    /**
     * Выполняет градиентный спуск для решения системы Ax = b и строит график норм резидуалов.
     *
     * @param data     Дополненная матрица [A | b].
     * @param tol      Точность для критерия сходимости.
     * @param maxIters Максимальное количество итераций.
     * @return Приближённое решение в виде массива.
     */
    public static double[] gradientDescentWithPlot(double[][] data, double tol, int maxIters) {
        // Разделяем дополненную матрицу на матрицу коэффициентов A и вектор правых частей b
        int rows = data.length;
        int cols = data[0].length;
        double[][] aData = new double[rows][cols - 1];
        double[] bData = new double[rows];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, aData[i], 0, cols - 1);
            bData[i] = data[i][cols - 1];
        }

        RealMatrix A = MatrixUtils.createRealMatrix(aData);
        RealVector b = MatrixUtils.createRealVector(bData);

        // Начальное приближение (нулевой вектор)
        RealVector x = new ArrayRealVector(A.getColumnDimension());

        // Шаговая матрица: A * A^T
        RealMatrix stepMatrix = A.multiply(A.transpose());

        // Список для хранения норм резидуалов
        List<Double> residualNorms = new ArrayList<>();

        boolean converged = false;
        int iteration;
        for (iteration = 0; iteration < maxIters; iteration++) {
            // Вычисляем остаток: r = A * x - b
            RealVector residual = A.operate(x).subtract(b);

            // Вычисляем шаг: step = (r^T * stepMatrix * r) / ( (stepMatrix * r)^T * (stepMatrix * r) )
            RealVector temp = stepMatrix.operate(residual);
            double numerator = residual.dotProduct(temp);
            double denominator = temp.dotProduct(temp);

            if (denominator == 0) {
                System.out.println("Знаменатель стал равен нулю. Остановка итераций для избежания деления на ноль.");
                break;
            }

            double step = numerator / denominator;

            // Обновляем решение: x_new = x - step * A^T * r
            RealVector xNew = x.subtract(A.transpose().operate(residual).mapMultiply(step));

            // Вычисляем норму разницы для критерия остановки
            double diffNorm = x.subtract(xNew).getNorm();
            residualNorms.add(residual.getNorm());

            // Обновляем текущее решение
            x = xNew;

            // Проверяем сходимость
            if (diffNorm < tol) {
                System.out.println("Сходимость достигнута на итерации " + (iteration + 1));
                converged = true;
                break;
            }
        }

        if (!converged) {
            System.out.println("Градиентный спуск не сошелся за максимальное количество итераций.");
        }

        // Строим график норм резидуалов
        plotResidualNorms(residualNorms);

        // Вычисляем и выводим невязку для найденного решения
        RealVector finalResidual = A.operate(x).subtract(b);
        System.out.println("Невязка ||Ax - b||: " + finalResidual.getNorm());

        return x.toArray();
    }

    /**
     * Строит график норм резидуалов с использованием JFreeChart.
     *
     * @param residualNorms Список норм резидуалов, записанных на каждой итерации.
     */
    private static void plotResidualNorms(List<Double> residualNorms) {
        XYSeries series = new XYSeries("Норма Резидуала");

        for (int i = 0; i < residualNorms.size(); i++) {
            // Используем log10 номера итерации (i+1) по оси X
            series.add(Math.log10(i + 1), residualNorms.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Сходимость Градиентного Спуска",
                "Log10(Номер Итерации)",
                "Норма Резидуала ||Ax - b||",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Отображаем график в окне JFrame
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Нормы Резидуалов Градиентного Спуска");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new ChartPanel(chart));
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    public static void main(String[] args) {
        // Пример использования
        double[][] dataMatrix = {
                {3.82, 1.02, 0.75, 0.81, 15.655},
                {1.05, 4.53, 0.98, 1.53, 22.705},
                {0.73, 0.85, 4.71, 0.81, 23.480},
                {0.88, 0.81, 1.28, 3.50, 16.110}
        };

        double tolerance = 1e-5;
        int maximumIterations = 1000;

        double[] result = gradientDescentWithPlot(dataMatrix, tolerance, maximumIterations);
        System.out.print("Приближённое решение: ");
        printArray(result);

        double[] originalAnswer = {2.12727865, 3.03258813, 3.76611894, 1.98884747};
        System.out.print("Изначальный ответ: ");
        printArray(originalAnswer);

        // Вычисляем и выводим невязку для исходного ответа
        RealMatrix A = MatrixUtils.createRealMatrix(new double[][]{
                {3.82, 1.02, 0.75, 0.81},
                {1.05, 4.53, 0.98, 1.53},
                {0.73, 0.85, 4.71, 0.81},
                {0.88, 0.81, 1.28, 3.50}
        });
        RealVector b = MatrixUtils.createRealVector(new double[]{15.655, 22.705, 23.480, 16.110});
        RealVector originalX = MatrixUtils.createRealVector(originalAnswer);
        RealVector originalResidual = A.operate(originalX).subtract(b);
        System.out.println("Невязка для изначального ответа ||Ax - b||: " + originalResidual.getNorm());
    }

    /**
     * Вспомогательный метод для печати массива чисел с плавающей точкой.
     *
     * @param array Массив для печати.
     */
    private static void printArray(double[] array) {
        System.out.print("[");
        for (int i = 0; i < array.length; i++) {
            System.out.printf("%.8f", array[i]);
            if (i < array.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");
    }
}
