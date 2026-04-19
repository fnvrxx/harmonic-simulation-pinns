classdef pinn_train < handle
    properties
        model
        lossValues
        learnRate
        avgGrad
        avgSqGrad
        iteration
    end

    methods
        function obj = pinn_train(layers, learnRate)
            obj.model      = dlnetwork(layers);
            obj.lossValues = [];
            obj.learnRate  = learnRate;
            obj.avgGrad    = [];
            obj.avgSqGrad  = [];
            obj.iteration  = 0;
        end

        function obj = trainStep(obj, t_data, u_data, t_pinn, m, mu, k)
            [loss, gradients] = dlfeval(@modelGradients, obj.model, t_data, u_data, t_pinn, m, mu, k);

            obj.iteration = obj.iteration + 1;
            [obj.model, obj.avgGrad, obj.avgSqGrad] = adamupdate( ...
                obj.model, gradients, obj.avgGrad, obj.avgSqGrad, obj.iteration, obj.learnRate);

            obj.lossValues(end+1) = extractdata(loss);
        end
    end
end
