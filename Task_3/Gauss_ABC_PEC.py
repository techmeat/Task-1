from tools import *
"* импорт с получением доступа ко всем именам"
from Animation import Probe, Source, AnimateFieldDisplay
import numpy.typing as npt
from typing import List
import matplotlib.pyplot as plt
import scipy.constants as sycon

def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma

def showProbeSignals(probes: List[Probe], minYSize: float, maxYSize: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''
    # Создание окна с графиков
    fig, ax = plt.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E) * dt * 1e9)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('t, нс')
    ax.set_ylabel('Ez, В/м')
    ax.grid()
    time_list = np.arange(len(probes[0].E)) * dt * 1e9
    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(time_list, probe.E)

    # Показать окно с графиками
    plt.show()

if __name__ == '__main__':
    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = sycon.c

    # Диэлектрическая постоянная
    eps0 = 8.854187817e-12

    # Параметры моделирования
    # Частота сигнала, Гц
    f_Hz = 0.5e9

    # Дискрет по пространству в м
    dx = 2e-3

    wavelength = c / f_Hz
    Nl = wavelength / dx

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 3

    # Время расчета в секундах
    maxTime_s = 47e-9

    # Положение источника в м
    sourcePos_m = 1.5

    # Координаты датчиков для регистрации поля в м
    probesPos_m = [0.5]

    # Параметры слоев
    layers_cont = [LayerContinuous(xmin = 0, eps = 7.0, sigma = 0.0)]

    # Скорость обновления графика поля
    speed_refresh = 15

    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    # Вывод параметров моделирования
    print(f'Число Куранта: {Sc}')
    print(f'Размер области моделирования: {maxSize_m} м')
    print(f'Время расчета: {maxTime_s * 1e9} нс')
    print(f'Координата источника: {sourcePos_m} м')
    print(f'Частота сигнала: {f_Hz * 1e-9} ГГц')
    print(f'Длина волны: {wavelength} м')
    print(f'Количество отсчетов на длину волны (Nl): {Nl}')
    probes_m_str = ', '.join(['{:.6f}'.format(pos) for pos in probesPos_m])
    print(f'Дискрет по пространству: {dx} м')
    print(f'Дискрет по времени: {dt * 1e9} нс')
    print(f'Координата пробника [м]: {probes_m_str}')
    print()
    print(f'Размер области моделирования: {maxSize} отсч.')
    print(f'Время расчета: {maxTime} отсч.')
    print(f'Координата источника: {sourcePos} отсч.')
    probes_str = ', '.join(['{}'.format(pos) for pos in probesPos])
    print(f'Координата пробника [отсч.]: {probes_str}')

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Коэффициенты для учета потерь
    loss = sigma * dt / (2 * eps * eps0)
    ceze = (1.0 - loss) / (1.0 + loss)
    cezh = W0 / (eps * (1.0 + loss))

    # Расчет коэффициентов для граничных условий
    # Sc' для левой границы
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)


    

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)



    # Источник
    magnitude = 1.0
    signal = Gaussian(1,300,40)

    source = Source(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)



    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения
        Hy[sourcePos - 1] += source.getH(t)


        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1: -1] * Ez[1: -1] + cezh[1: -1] * (Hy[1:] - Hy[: -1])

        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]





        # Источник возбуждения
        Ez[sourcePos] += source.getE(t)

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)
            display.stop()
    # Отображение сигнала, сохраненного в пробнике
    showProbeSignals(probes, -2.1, 2.1)

    # Построение спектра
    plt.figure()
    sp  = np.fft.fft(probes[0].E)
    freq = np.fft.fftfreq(maxTime)
    plt.plot(freq / (dt * 1e9), abs(sp) / max(abs(sp)))
    plt.xlim(0, 2.5)
    plt.grid()
    plt.xlabel('f, ГГц')
    display.stop()
    plt.show()
