#possible v4
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import librosa
import IPython.display as ipd
import soundfile as sf
from sklearn.cluster import KMeans
from beat_this.inference import File2Beats
import os
def test_bpm(
        audio_path = "/home/acacia/下载/名乗る名が無い - Designant. (Game ver.).flac",
        checkpoint_path = "final0",
        dbn = False,
        min_interval = 0.15,  # 最小间隔
        max_interval = 1.0,   # 最大间隔
        kernel_size=11,  # soomth_kernel, 必须是奇数
        tolerance=4.0,  # BPM差异容差
        min_duration=2,  # 最小持续时间
        output_path='./result/',

):
    os.mkdir(output_path)
    file2beats = File2Beats(checkpoint_path="final0", dbn=False)
    origin_beats, _ = file2beats(audio_path)

    # 1. 剔除异常节拍点
    beats = np.array(origin_beats)
    intervals = np.diff(beats)

    valid_indices = np.where((intervals >= min_interval) & (intervals <= max_interval))[0]
    filtered_beats = np.array([beats[0]] + [beats[i+1] for i in valid_indices])
    filtered_intervals = np.diff(filtered_beats)

    # 2. 计算原始 BPM
    raw_bpms = 60 / filtered_intervals

    # 3. 识别真正的基准BPM
    # 创建BPM直方图
    plt.figure(figsize=(12, 4))
    hist, bins = np.histogram(raw_bpms, bins=100, range=(60, 600))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.bar(bin_centers, hist, width=bins[1]-bins[0])
    plt.title("BPM distribution")
    plt.xlabel("BPM")
    plt.ylabel("frequency")
    plt.grid(True)
    plt.savefig(output_path + 'BPM_distribution.png', dpi=300, bbox_inches='tight')  # 保存图形
    # plt.show()

    # 使用聚类找出主要BPM组
    # 限制聚类数为5，涵盖可能的1x, 2x, 3x, 4x倍速
    n_clusters = min(5, len(raw_bpms))
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(raw_bpms.reshape(-1, 1))
    centers = sorted(km.cluster_centers_.flatten())

    print("检测到的BPM聚类中心:", [round(c) for c in centers])

    # 寻找可能的基准BPM（查找成倍数关系）
    ratios = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            ratio = centers[j] / centers[i]
            ratios.append((ratio, centers[i], centers[j]))

    # 筛选接近整数倍的比例
    integer_ratios = [(r, base, higher) for r, base, higher in ratios
                      if abs(round(r) - r) < 0.15 and round(r) > 1]

    if integer_ratios:
        # 按基准BPM从小到大排序
        integer_ratios.sort(key=lambda x: x[1])
        base_bpm = integer_ratios[0][1]
        print(f"推测基准BPM为: {round(base_bpm)}")
    else:
        # 如果没找到明显的倍数关系，使用最小的聚类中心
        base_bpm = centers[0]
        print(f"未检测到明显倍数关系，使用最小值作为基准BPM: {round(base_bpm)}")

    # 4. 将所有BPM归一化到基准
    normalized_bpms = np.copy(raw_bpms)
    for i in range(len(normalized_bpms)):
        ratio = normalized_bpms[i] / base_bpm
        if ratio > 1.3:  # 如果是基准的1.3倍以上
            # 找到最接近的整数倍
            multiple = round(ratio)
            if multiple > 1 and abs(ratio - multiple) < 0.2:
                normalized_bpms[i] = normalized_bpms[i] / multiple

    # 5. 平滑处理

    smoothed_bpms = medfilt(normalized_bpms, kernel_size)

    # 6. 识别稳定BPM段落
    stable_bpms = []
    stable_start_times = []
    stable_end_times = []
    stable_durations = []


    current_bpm = smoothed_bpms[0]
    segment_start_idx = 0
    time_points = filtered_beats[:-1]

    for i in range(1, len(smoothed_bpms)):
        if abs(smoothed_bpms[i] - current_bpm) > tolerance:
            duration = time_points[i-1] - time_points[segment_start_idx]
            if duration >= min_duration:
                stable_bpms.append(round(current_bpm))
                stable_start_times.append(time_points[segment_start_idx])
                stable_end_times.append(time_points[i-1])
                stable_durations.append(duration)
            current_bpm = smoothed_bpms[i]
            segment_start_idx = i

    # 处理最后一个段落
    duration = time_points[-1] - time_points[segment_start_idx]
    if duration >= min_duration:
        stable_bpms.append(round(current_bpm))
        stable_start_times.append(time_points[segment_start_idx])
        stable_end_times.append(time_points[-1])
        stable_durations.append(duration)

    # 7. 调整原始节拍点并处理倍速区域
    corrected_beats = []
    last_beat_time = -np.inf

    for beat in origin_beats:
        # 查找当前节拍所在的稳定BPM段落
        current_segment = -1
        for i in range(len(stable_bpms)):
            if stable_start_times[i] <= beat <= stable_end_times[i]:
                current_segment = i
                break

        if current_segment >= 0:
            segment_bpm = stable_bpms[current_segment]
            expected_interval = 60 / segment_bpm

            # 检查与上一个节拍的时间间隔
            if len(corrected_beats) > 0:
                last_beat = corrected_beats[-1]
                interval = beat - last_beat

                # 如果间隔太小(说明可能是高倍速区域的额外节拍)
                if interval < 0.8 * expected_interval:
                    # 跳过这个节拍点(不添加到corrected_beats)
                    continue

                # 检查是否缺失节拍
                if interval > 1.5 * expected_interval:
                    # 添加缺失的节拍点
                    num_missing = round(interval / expected_interval) - 1
                    for i in range(1, num_missing + 1):
                        missing_beat = last_beat + i * expected_interval
                        corrected_beats.append(missing_beat)

                # 如果间隔合理或接近预期间隔的整数倍，保留
                if (abs(interval % expected_interval) < 0.2 * expected_interval or
                    abs(interval % expected_interval - expected_interval) < 0.2 * expected_interval):
                    corrected_beats.append(beat)
                else:
                    # 调整到最接近的正确位置
                    expected_beat = last_beat + round(interval / expected_interval) * expected_interval
                    if abs(beat - expected_beat) < 0.3 * expected_interval:
                        corrected_beats.append(expected_beat)
            else:
                # 第一个节拍点
                corrected_beats.append(beat)
        else:
            # 不在稳定段落内，直接添加
            corrected_beats.append(beat)

    # 8. 可视化
    plt.figure(figsize=(14, 6))
    plt.plot(time_points, raw_bpms, 'r.', alpha=0.3, label='raw_BPM')
    plt.plot(time_points, normalized_bpms, 'g.', alpha=0.3, label='normalized_BPM')
    plt.plot(time_points, smoothed_bpms, 'b-', linewidth=2, label='soomthed_BPM')

    # 标记稳定BPM段落
    for i in range(len(stable_bpms)):
        plt.axhline(y=stable_bpms[i], xmin=stable_start_times[i]/time_points[-1],
                    xmax=stable_end_times[i]/time_points[-1],
                    color='g', linestyle='-', linewidth=3, alpha=0.5)
        plt.text(stable_start_times[i]+stable_durations[i]/2, stable_bpms[i]+2,
                 f"{stable_bpms[i]} BPM", ha='center')

    plt.title(f"BPM change as time (base BPM: {round(base_bpm)})")
    plt.xlabel("time (s)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path + 'bpm_time_evolution.png', dpi=300, bbox_inches='tight')  # 保存图形
    # plt.show()

    # 9. 播放修正后的节拍
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio_beat = librosa.clicks(times=corrected_beats, sr=sr, click_freq=1000, length=len(audio))
    sf.write(output_path + "output_" + audio_path.split("/")[-1], audio + audio_beat, sr)
    # ipd.display(ipd.Audio(audio + audio_beat, rate=sr))

    # 打印结果统计
    print("\n稳定BPM段落统计 (基准BPM: {:.0f}):".format(base_bpm))
    print("-------------------------")
    print("BPM值 | 开始时间(秒) | 持续时间(秒)")
    print("-------------------------")
    for i in range(len(stable_bpms)):
        print(f"{stable_bpms[i]:4d} | {stable_start_times[i]:11.1f} | {stable_durations[i]:10.1f}")

    print(f"\n原始节拍数量: {len(origin_beats)}")
    print(f"修正后节拍数量: {len(corrected_beats)}")

test_bpm()
