import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.NeuralGAM.ngam import NeuralGAM
from src.utils.utils import generate_homoskedastic_uniform_data, plot_list, plot_partial_dependencies
import statsmodels.api as sm

if __name__ == "__main__":
      
    """ GET DATA - homoskedastic uniform """
    X_train, X_test, y_train, y_test  = generate_homoskedastic_uniform_data(nrows=25000, output_folder="./test/data/homoskedastic_uniform_/")
       
    # run statsmodel
    X_constant = sm.add_constant(X_train)
    lin_reg = sm.OLS(y_train, X_constant).fit()
    print(lin_reg.summary())  
    
    ngam = NeuralGAM(num_inputs = len(X_train.columns), num_units=64)
    ycal, mse = ngam.fit(X_train = X_train, y_train = y_train, max_iter = 20)
    
    ngam.save_model("./results/homoskedastic_uniform/model.ngam")
    
    print("Achieved RMSE during training = {0}".format(mean_squared_error(y_train, ngam.y, squared=False)))
        
    y_pred = ngam.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    
    training_fs = ngam.get_partial_dependencies(X_train)
    test_fs = ngam.get_partial_dependencies(X_test)
    
    plot_list([y_test, y_pred], ["real", "predicted"], title="MSE on prediction = {0}".format(mse))
    plot_partial_dependencies(X_test, test_fs, title="Prediction f(x)")
    plot_partial_dependencies(X_train, training_fs, title="Learned f(x) from training")
    plt.show(block=True)