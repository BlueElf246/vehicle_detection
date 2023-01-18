from load_dataset import *
from setting import params
os.chdir("/Users/datle/Desktop/car_detection")
car, non_car= load_dataset()
# car=car[:3]
# non_car=non_car[:3]
car_feature=extract_feature(car, 'RGB')
non_car_feature= extract_feature(non_car, 'RGB')
X,y= combine(car_feature, non_car_feature)
sc, X_scaled= normalize(X)
X_train, X_test, y_train, y_test= split(X_scaled, y)
model= train_model(X_train, X_test, y_train, y_test)
save_model('lp_detect.p', model, sc, params=params,y=y)
