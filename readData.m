function [train, test] = readData(x)

    switch x   
        case 'diabetic'
            data = table2array(readtable('diabetic.xlsx'));
            test = data(1:346,:);
            train = data(346:end,:);
        case 'cancer'
            data = table2array(readtable('cancer.xls'));
            test = data(1:172,:);
            train = data(172:end,:);
        case 'dota'
            train = table2array(readtable('dota2Train.csv'));
            test = table2array(readtable('dota2Test.csv'));
            train = train(1:10000,:);
            test = test(1:3000,:);
        case 'extention'
            data = table2array(readtable('extention.xlsx'));
            test = data(1:92,:);
            train = data(92:end,:);
        case 'vehicle'
            data = table2array(readtable('vehicle.xlsx'));
            test = data(1:255,:);
            train = data(255:end,:);
        case 'wine'
            data = table2array(readtable('wine.xlsx'));
            test = data(1:54,:);
            train = data(54:end,:);
        case 'CMC'
            data = table2array(readtable('CMC.xlsx'));
            test = data(1:443,:);
            train = data(443:end,:);
        case 'glass'
            data = table2array(readtable('glass.xlsx'));
            test = data(1:65,:);
            train = data(65:end,:);
        case 'TAE'
            data = table2array(readtable('TAE.xlsx'));
            test = data(1:46,:);
            train = data(46:end,:);
        case 'ionosphere'
            data = table2array(readtable('ionosphere.xlsx'));
            test = data(1:106,:);
            train = data(106:end,:);
        otherwise
        warning('Please check the dataset name')
    end
    
end