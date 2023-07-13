import numpy as np
from sklearn.linear_model import LinearRegression

data_metrics = np.array([
    [1005, 1005, 265, 285, 390, 471, 470, 430, 243, 252, 567, 564, 295, 295, 632, 382, 425, 365, 449, 469, 548, 574, 353, 343],
    [0, 745, 441, 455, 507, 579, 505, 555, 300, 305, 645, 635, 405, 405, 730, 460, 520, 460, 465, 485, 475, 445, 415, 405],
    [1011, 1011, 255, 265, 367, 439, 443, 435, 211, 220, 535, 532, 265, 265, 600, 350, 410, 350, 443, 463, 525, 544, 340, 330],
    [1204, 1204, 706, 736, 272, 346, 583, 589, 355, 364, 515, 465, 483, 483, 507, 377, 533, 473, 672, 685, 765, 735, 484, 474],
    [1023, 1023, 514, 533, 262, 343, 331, 302, 176, 181, 520, 540, 281, 281, 542, 272, 297, 237, 451, 412, 582, 595, 225, 215],
    [973, 973, 493, 492, 334, 406, 332, 358, 124, 0, 448, 468, 232, 232, 533, 263, 323, 263, 468, 488, 533, 523, 242, 232],
    [1120, 1120, 605, 621, 386, 464, 489, 505, 271, 280, 368, 280, 389, 389, 625, 410, 470, 410, 611, 631, 455, 448, 399, 389],
    [1101, 1101, 594, 613, 461, 533, 459, 490, 254, 259, 580, 600, 0, 120, 665, 395, 455, 395, 539, 503, 488, 480, 369, 359],
    [1861, 1861, 1383, 1385, 1116, 1197, 1225, 1156, 1012, 1021, 1336, 1330, 1125, 1125, 587, 317, 265, 0, 1327, 1266, 1426, 1411, 811, 801],
    [831, 831, 400, 408, 320, 392, 390, 405, 171, 180, 495, 511, 210, 210, 553, 310, 370, 310, 260, 280, 465, 440, 220, 210],
    [950, 950, 561, 571, 446, 522, 448, 470, 236, 245, 488, 400, 300, 300, 645, 375, 435, 375, 545, 547, 250, 250, 358, 348]
])

expected_values = np.array([
    [0.692, 0.818, 0.438, 0.422, 15.4, 13.4, 7.51, 10, 7.86, 7.48, 17.1, 13.3, 7.04, 7.04, 23.6, 21.1, 23.6, 21.4, 1.32, 1.34, 7.37, 11.2, 15.4, 11.9],
    [0.252, 0.595, 1.18, 1.31, 16, 16.1, 6.86, 7.81, 7.09, 7.08, 16.9, 17.1, 13.5, 14.4, 22.3, 20.5, 24.5, 24.5, 1.28, 1.37, 10.6, 10.4, 12.5, 12.5],
    [1.1, 1.04, 0.386, 0.508, 15, 14.8, 8.19, 7.07, 6.52, 8.66, 16, 11.9, 11.6, 14.1, 23.3, 22.1, 20.8, 22.3, 0.77, 0.795, 11.6, 12.8, 14.9, 15.2],
    [14.1, 15.5, 14.6, 14.7, 0.516, 0.504, 8.2, 8.43, 7.43, 8.17, 9.14, 9.13, 16.7, 15.8, 7.76, 7.84, 15.1, 14.8, 15.5, 13.3, 16.9, 16.9, 14.9, 14.9],
    [6.69, 6.7, 6.21, 7.81, 7.42, 7.11, 0.581, 0.626, 0.303, 0.28, 9.65, 8.29, 7.18, 8.09, 16.1, 14.2, 15.4, 15.4, 7.64, 7.57, 10.4, 9.93, 5.84, 5.85],
    [7.09, 7.56, 8.42, 6.19, 9.59, 7.71, 0.459, 0.463, 0.223, 0.382, 9.57, 8.68, 8.04, 8.93, 14.9, 13.2, 17.2, 17.2, 7.81, 9.67, 9.91, 9.43, 6.3, 6.31],
    [18.6, 16.7, 11.9, 13.5, 7.78, 8.66, 8.49, 7.77, 7.54, 7.51, 0.463, 0.437, 17, 17, 16.1, 17.6, 21.9, 21.6, 14.1, 14.2, 11.3, 10.2, 14.7, 16],
    [13.5, 13.7, 7.13, 6, 16.3, 18.1, 8.26, 6.33, 8.05, 8.05, 16.3, 14.5, 0.703, 0.224, 20.4, 20, 20.4, 21.6, 4.94, 3.16, 7.26, 6.98, 20.2, 14.3],
    [24.5, 24.5, 26.9, 21.4, 22.5, 22.2, 14.9, 15.5, 17.2, 17.1, 26.5, 24, 21.4, 21.4, 4.65, 4.57, 0.224, 0.402, 22.7, 22.7, 24.3, 26.4, 17.6, 17.6],
    [0.909, 0.896, 0.665, 0.947, 16, 13.2, 6.32, 6.18, 6.75, 6.1, 16.1, 18.7, 4.31, 4.32, 20.3, 20.7, 19.9, 19.7, 0.615, 0.73, 7.77, 8.04, 11.9, 11.5],
    [10.4, 10, 7.64, 11.7, 16.3, 19.3, 15, 7.22, 6.95, 6.93, 8.32, 12.8, 6.68, 6.68, 21.9, 19.9, 23.9, 23.9, 7.83, 7.67, 0.406, 0.524, 16.4, 19.3]
])

matching_scores = np.zeros_like(expected_values)

for i in range(len(expected_values)):
    for j in range(len(expected_values[0])):
        if expected_values[i][j] in data_metrics:
            matching_scores[i][j] = 1
        else:
            matching_scores[i][j] = np.sum(np.abs(data_metrics - expected_values[i][j]))

min_indices = np.argmin(matching_scores, axis=1)
metrics_values = np.zeros(len(expected_values))
for i in range(len(expected_values)):
    if expected_values[i][min_indices[i]] in data_metrics:
        metrics_values[i] = expected_values[i][min_indices[i]]
    else:
        min_unmatched_value = np.min(matching_scores[i])
        min_unmatched_index = np.argmin(matching_scores[i])
        metrics_values[i] = data_metrics[min_unmatched_index // len(data_metrics[0])][min_unmatched_index % len(data_metrics[0])]

regression_model = LinearRegression()
regression_model.fit(data_metrics.flatten().reshape(-1, 1), expected_values.flatten().reshape(-1, 1))
predicted_metrics_values = regression_model.predict(metrics_values.reshape(-1, 1))
print(predicted_metrics_values.flatten())
