import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

def expected_time_two_elevators(n, f, h, v, p_stops1, p_stops2, t_stop):
    total = 0
    for k1 in range(1, n + 1):
        T1 = abs(k1 - f) * h / v + sum(p_stops1[min(k1, f)+1:max(k1, f)]) * t_stop
        for k2 in range(1, n + 1):
            T2 = abs(k2 - f) * h / v + sum(p_stops2[min(k2, f)+1:max(k2, f)]) * t_stop
            total += min(T1, T2)
    return total / (n * n)

def monte_carlo_two_elevators(n, f, h, v, p_stops1, p_stops2, t_stop, sims=10000):
    times = []
    floors = np.arange(1, n + 1)
    for _ in range(sims):
        k1, k2 = np.random.choice(floors, 2)
        stops1 = sum(np.random.rand(len(range(min(k1,f)+1, max(k1,f)))) < 
                     np.array(p_stops1[min(k1, f)+1:max(k1, f)]))
        T1 = abs(k1 - f) * h / v + stops1 * t_stop
        stops2 = sum(np.random.rand(len(range(min(k2,f)+1, max(k2,f)))) < 
                     np.array(p_stops2[min(k2, f)+1:max(k2, f)]))
        T2 = abs(k2 - f) * h / v + stops2 * t_stop
        times.append(min(T1, T2))
    return np.mean(times)

def expected_descent_time(f, h, v, p_stops, t_stop):
    return abs(f - 1) * h / v + sum(p_stops[2:f]) * t_stop

def monte_carlo_descent_time(f, h, v, p_stops, t_stop, sims=10000):
    times = []
    for _ in range(sims):
        stops = sum(np.random.rand(f-2) < np.array(p_stops[2:f]))
        travel_time = abs(f - 1) * h / v + stops * t_stop
        times.append(travel_time)
    return np.mean(times)

def calculate_vika_travel_time(distance, speed):
    return distance / speed

def train_time(speed, distance):
    return distance / (speed * 1000/3600)

st.title("Успеет ли Вика на поезд? 🚆")

# Инициализация вероятностей остановок
if 'p_stops1' not in st.session_state or len(st.session_state.p_stops1) != 21:
    st.session_state.p_stops1 = [0.0] * 21
if 'p_stops2' not in st.session_state or len(st.session_state.p_stops2) != 21:
    st.session_state.p_stops2 = [0.0] * 21

with st.sidebar.expander("🔧 Основные параметры задачи", expanded=True):
    n = st.number_input("Число этажей", 3, 20, 9)
    f = st.number_input("Этаж вызова", 1, n, min(8, n))
    h = st.number_input("Высота этажа (м)", 2.0, 5.0, 3.0, step=0.1)
    v = st.number_input("Скорость лифта (м/с)", 0.5, 5.0, 1.3, step=0.1)
    constant_delay = st.number_input("Константная задержка (сек)", 0, 30, 5)
    t_stop = st.number_input("Задержка остановки (сек)", 5, 60, 15)

with st.sidebar.expander("🚶 Путь до станции"):
    distance_to_station = st.number_input("До станции (м)", 100, 2000, 176)
    vika_avg_speed = st.number_input("Скорость Вики (м/с)", 0.5, 25.0, 2.0, step=0.1)
    time_to_validation = st.number_input("Валидация билета (сек)", 0, 30, 5)

with st.sidebar.expander("🚂 Параметры поезда"):
    avg_train_speed = st.number_input("Скорость поезда (км/ч)", 5, 200, 40, step=5)
    train_distance_to_station = st.number_input("Путь поезда (м)", 100, 2000, 850)
    train_stop_time = st.number_input("Стоянка поезда (сек)", 0, 240, 30)

with st.sidebar.expander("🎲 Симуляции и опции"):
    num_simulations = st.number_input("Кол-во симуляций", 1000, 50000, 10000, step=1000)
    use_stairs = st.checkbox("Идти по лестнице")
    stairs_time_per_floor = st.number_input("Секунд на этаж (лестница)", 1, 60, 10)

with st.sidebar.expander("📊 Вероятности остановок по этажам"):
    p_stops1, p_stops2 = st.session_state.p_stops1, st.session_state.p_stops2
    for floor in range(2, int(f)):
        col1, col2 = st.columns(2)
        with col1:
            p_stops1[floor] = st.slider(f"Лифт 1, этаж {floor}", 0.0, 1.0, p_stops1[floor], 0.05, key=f"l1_{floor}")
        with col2:
            p_stops2[floor] = st.slider(f"Лифт 2, этаж {floor}", 0.0, 1.0, p_stops2[floor], 0.05, key=f"l2_{floor}")

# Расчёты
train_full_time = train_time(avg_train_speed, train_distance_to_station) + train_stop_time
vika_run_time = calculate_vika_travel_time(distance_to_station, vika_avg_speed)

if use_stairs:
    descent_time = (f - 1) * stairs_time_per_floor
    total_time = descent_time + constant_delay + vika_run_time + time_to_validation
    diff = total_time - train_full_time
    results = pd.DataFrame({
        "Показатель": [
            "Спуск (лестница)",
            "Константная задержка",
            "Бег до станции",
            "Суммарное время поезда",
            "Общее время (лестница)",
            "Разница (Вика - поезд)"
        ],
        "Секунды": [
            f"{descent_time:.2f}",
            f"{constant_delay:.2f}",
            f"{vika_run_time:.2f}",
            f"{train_full_time:.2f}",
            f"{total_time:.2f}",
            f"{diff:.2f}"
        ]
    })
    final_time = total_time
else:
    lift_mc = monte_carlo_two_elevators(n, f, h, v, p_stops1, p_stops2, t_stop, int(num_simulations))
    descent_mc = monte_carlo_descent_time(f, h, v, p_stops1, t_stop, int(num_simulations))
    total_time_mc = lift_mc + descent_mc + constant_delay + vika_run_time + time_to_validation
    diff = total_time_mc - train_full_time
    results = pd.DataFrame({
        "Показатель": [
            "Лифт и прибытие (Монте-Карло)",
            "Спуск (Монте-Карло)",
            "Константная задержка",
            "Бег до станции",
            "Суммарное время поезда",
            "Общее время (Монте-Карло)",
            "Разница (Вика - поезд)"
        ],
        "Секунды": [
            f"{lift_mc:.2f}",
            f"{descent_mc:.2f}",
            f"{constant_delay:.2f}",
            f"{vika_run_time:.2f}",
            f"{train_full_time:.2f}",
            f"{total_time_mc:.2f}",
            f"{diff:.2f}"
        ]
    })
    final_time = total_time_mc

st.table(results.set_index("Показатель"))

if final_time <= train_full_time:
    st.success("✅ Успеет!")
else:
    st.error("❌ Опоздает!")
