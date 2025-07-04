import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'gio_hoc': [1, 2, 3, 4, 5, 6, 7],
    'diem_thi': [40, 45, 50, 55, 60, 65, 70]
}

df = pd.DataFrame(data)
X = df[['gio_hoc']]
y = df[['diem_thi']]

model = LinearRegression()  
model.fit(X, y)

du_doan_moi = pd.DataFrame({'gio_hoc': [8]})
diem_thi_doan = model.predict(du_doan_moi)

print('Dự đoán điểm nếu học 8 giờ là:', round(diem_thi_doan[0][0], 2))

# draw
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy')
plt.xlabel('Giờ học')
plt.ylabel('Điểm thi')
plt.title('AI dự đoán điểm thi toán')
plt.legend()
plt.show()
