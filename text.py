import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import streamlit as st

st.sidebar.title('202302805박민재')
show_k_means = st.sidebar.checkbox("k-means 클러스터링")
show_step_response = st.sidebar.checkbox('폐루프 전달함수 Unit step 응답 곡선')
show_bode_diagram = st.sidebar.checkbox('폐루프 전달함수 주파수 응답 곡선')


def plot_step_response():
    # 전달함수의 분자와 분모 계수
    numerator = [100]
    denominator = [1, 5, 6]

    # 폐루프 전달함수 계산
    closed_loop_tf = signal.TransferFunction(numerator, denominator)

    # 시간 범위 설정
    t = np.linspace(0, 10, 1000)

    # Unit Step 입력 생성
    u = np.ones_like(t)

    # 시스템 응답 계산
    _, y = signal.step(closed_loop_tf, T=t)

    # 응답 곡선 그리기
    plt.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Step Response')
    plt.grid(True)
    st.pyplot()

def plot_bode_diagram():
    # 전달함수의 분자와 분모 계수
    numerator = [100]
    denominator = [1, 5, 6]

    # 폐루프 전달함수 계산
    closed_loop_tf = signal.TransferFunction(numerator, denominator)

    # 주파수 응답 계산
    w, mag, phase = signal.bode(closed_loop_tf)

    # 보드선도 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.semilogx(w, mag)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Bode Diagram')
    ax1.grid(True)

    ax2.semilogx(w, phase)
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)
    st.pyplot()

if show_k_means:
    data = np.random.randn(100, 2)

    def initialize_centroids(data, k):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        return centroids

    def assign_clusters(data, centroids):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        return clusters

    def update_centroids(data, clusters, k):
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        return new_centroids

    def k_means(data, k, max_iterations=100):
        centroids = initialize_centroids(data, k)
        prev_centroids = centroids.copy()

        for _ in range(max_iterations):
            clusters = assign_clusters(data, centroids)
            centroids = update_centroids(data, clusters, k)

            if np.all(prev_centroids == centroids):
                break
            prev_centroids = centroids.copy()
        return prev_centroids, clusters

    k = 3
    centroids, clusters = k_means(data, k)

    # Create a Streamlit figure
    fig, ax = plt.subplots()
    for i in range(k):
        cluster_data = data[clusters == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i+1}")

    ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", color="k", s=100, label="Centroids")
    ax.legend()
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("k-means clustering")
    st.pyplot(fig)

# Streamlit 애플리케이션 시작

if show_step_response:
    st.header('폐루프 전달함수 응답 곡선')
    st.write('Unit Step 입력에 대한 응답 곡선을 표시합니다.')
    plot_step_response()

if show_bode_diagram:
    st.header('폐루프 전달함수 주파수 응답')
    st.write('주파수 응답을 보드선도로 표시합니다.')
    plot_bode_diagram()

if show_k_means:
    data = np.random.randn(100, 2)
    st.header('k_means')
    st.write('k-means clustering 표시합니다.')
    plot_bode_diagram()
