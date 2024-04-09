clear variables;
% calea către folder-ul cu imagini originale
folder = 'C:\Users\User\Desktop\Licenta\Licenta\HYTA-master\HYTA-master\images';
%calea către folder-ul cu imagini segmentate manual
folder2 = 'C:\Users\User\Desktop\Licenta\Licenta\HYTA-master\HYTA-master\2GT_img';

% obținere informații despre fișierele .jpg din folder și folder2
images = dir(fullfile(folder, '*.jpg'));
images2 = dir(fullfile(folder2, '*.jpg'));
imageNumber = length(images);

% memoreaza numele imaginilor într-un vector
numImages = numel(images);
imageNames = strings(numImages, 1);

for i = 1:numImages
    imageNames(i) = images(i).name;
end

FSC_values = zeros(imageNumber, 1);
CB_values = zeros(imageNumber, 1);
TH_values = zeros(imageNumber, 1);

for w = 1:imageNumber
    image_name = images(w).name;
    image_path = fullfile(images(w).folder, image_name);
    image = imread(image_path);     %importarea imaginii
    
image_name2 = images2(w).name;
    image2 = imread(fullfile(images2(w).folder, image_name2));

correct_segmented = im2double(image2);
bwCorrect_segmented = imbinarize(correct_segmented, 0.25);
 imshow(bwCorrect_segmented)

    % convert image to double precision
    image = im2double(image);
    
    red_channel = image(:, :, 1);  % extragerea canalului roșu
    blue_channel = image(:, :, 3);   %extragerea canalului albastru
    norm_ratio = (blue_channel - red_channel) ./ (blue_channel + red_channel);
    
%     % create a new figure to display the images
%     figure('Name', image_name);
%     
%    % display the original image
%     subplot(1, 4, 1);
%     imshow(image);
%     title('Original image');
%     
%     % display the red channel
%     subplot(1, 4, 2);
%     imshow(red_channel);
%     title('Red channel');
    
%     % display the blue channel
%     subplot(1, 4, 3);
%     imshow(blue_channel);
%     title('Blue channel');
    
%     % display the normalized ratio
%     subplot(1, 4, 4);
%     imshow(norm_ratio);
%     title('Normalized ratio');

disp(['Image name: ' image_name]);
[m, n] = size(norm_ratio);  %dimensiunea matricii
disp(['The matrix has ' num2str(m) ' rows and ' num2str(n) ' columns.']);
number_pixels = m*n; %numărul de pixeli ai imaginii
disp(['The image has ' num2str(number_pixels) ' pixels.']); 

%instantiere imagine noua, cu valori de 0, de dimensiune m*n
imagine_afisare = zeros(m, n, 'double'); %instantiere imagine noua, cu valori de 0, de dimensiune m*n

%deviatia standard
standardDeviation = std2(norm_ratio);  %deviația standard
disp(['Standard deviation: ' num2str(standardDeviation)]);
stdDev_values(w) = standardDeviation;

data = norm_ratio(:);  %transformarea matricii într-un vector de valori
unic_pix = unique(data);   %determinarea valorilor unice din vectorul de valori
counts = histc(data, unic_pix);   %calcularea histogramei în funcție de parametrii aleși
%plot(unic_pix, counts, 'b')   %afișarea histogramei

t_min = Inf;
CE_min = Inf;
CE_values = [];
L = numel(counts);

%determinam dacă o imagine este unimodală sau bimodală
if (standardDeviation>0.03) 
    disp("The image is bimodal");
    %algoritm prag MCE

num_mu1 = zeros(1, L);
den_mu1 = zeros(1, L);
num_mu2 = zeros(1, L);
den_mu2 = zeros(1, L);
term1 = zeros(1, L);
term2 = zeros(1, L);

for t = 1:L  %pentru fiecare bin/ratio level
     
    num_mu1(t) = 0;
    den_mu1(t) = 0;
    num_mu2(t) = 0;
    den_mu2(t) = 0;
    term1(t) = 0;
    term2(t) = 0;
    
    for i = 1:t  %pentru fiecare bin/ratio level de la 1 la t
        num_mu1(t) = num_mu1(t) + i*counts(i); %numarator
        den_mu1(t) = den_mu1(t) + counts(i); %numitor = denominator
    end

    mu1 = num_mu1(t)/den_mu1(t);

    for i = 1:t
        term1(t) = term1(t) + i*counts(i)*log(i/mu1);
    end

    for i = t+1:L
        num_mu2(t) = num_mu2(t) + i*counts(i);
        den_mu2(t) = den_mu2(t) + counts(i);
    end
    
    mu2 = num_mu2(t)/den_mu2(t);
    
    for i = t+1:L
        term2(t) = term2(t) + i*counts(i)*log(i/mu2);
    end

    CE = term1(t)+term2(t);
CE_values = [CE_values CE];

    if CE<CE_min
        CE_min = CE;
        t_min = unic_pix(t);
    end

end
bww2 = imbinarize(norm_ratio, t_min);
bimodal_segmented = 1 - bww2;
imagine_afisare = bimodal_segmented;
else
    disp("The image is unimodal");
    %algoritm prag fix
    
binary_mask = imbinarize(norm_ratio, 0.250);
unimodal_segmented = 1 - binary_mask;
imagine_afisare = unimodal_segmented;


end
% disp(['Imaginea segmentata rezultata']);
% subplot(1, 4, 2);
% imshow(imagine_afisare);
% title('Segmentare rezultata');



% disp(['Imaginea segmentata corect']);
% subplot(1, 4, 3);
% imshow(bwCorrect_segmented);
% title('Segmentare corecta');

cloudPixels = 0;
for k = 1:m
    for l = 1:n
        if imagine_afisare(k, l) == 1
            cloudPixels = cloudPixels + 1;
        end
    
    end
end

cloud_fraction = cloudPixels/number_pixels;
disp(['Cloud fraction: ' num2str(cloud_fraction)]);

TP = 0;
TN = 0;
FP = 0;
FN = 0;
for k = 1:m
    for l = 1:n
        if ((imagine_afisare(k, l) == bwCorrect_segmented(k, l)) && imagine_afisare(k,l) == 1)
          TP = TP + 1;
        end
        if((imagine_afisare(k, l) == bwCorrect_segmented(k, l)) && imagine_afisare(k, l) == 0)
            TN = TN + 1;
        end
        if((imagine_afisare(k, l) ~= bwCorrect_segmented(k, l)) && imagine_afisare(k, l) == 0)
            FP = FP + 1;
        end
        if((imagine_afisare(k, l) ~= bwCorrect_segmented(k, l)) && imagine_afisare(k, l) == 1)
            FN = FN + 1;
        end
    end
end

disp(['Confusion Matrix']);
confusionMatrix = [TP FP; FN TN]

if( TP == 0 && FP == 0 && FN ==0 && TN~=0)
Pr = 0; Rc = 0;Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP == 0 && FP ==0 && FN ~=0 && TN == 0)
        Pr = 0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP == 0 && FP == 0 && FN ~=0 && TN ~= 0)
        Pr = 0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP == 0 && FP ~= 0 && FN ==0 && TN == 0)
        Pr = 0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP ==0 && FP ~=0 && FN == 0 && TN ~= 0)
    Pr = 0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP == 0 && FP ~=0 && FN ~= 0 && TN == 0)
Pr = 0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ==0 && FP ~=0 && FN ~= 0 && TN~= 0)
    Pr =0; Rc = 0; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~=0 && FP ==0 && FN == 0 && TN == 0)
    Pr = 1; Rc = 1; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~=0 && FP == 0 && FN == 0 && TN ~=0)
    Pr = 1; Rc = 1; Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~= 0 && FP == 0 && FN ~=0 && TN == 0)
    Pr = 1; Rc = TP/(TP+FN); Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~= 0 && FP == 0 && FN ~=0 && TN ~=0)
    Pr = TP/(TP + FP); Rc = TP/(TP + FN); Ac = (TP + TN)/(TP + FP + TN + FN);
elseif (TP ~=0 && FP ~= 0 && FN == 0 && TN == 0)
     Pr = TP/(TP + FP); Rc = TP/(TP + FN); Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~=0 && FP ~=0 && FN == 0 && TN ~= 0)
    Pr = TP/(TP + FP); Rc = TP/(TP + FN); Ac = (TP + TN)/(TP + FP + TN + FN);
elseif(TP ~=0 && FP ~=0 && FN ~=0 && TN == 0)
    Pr = TP/(TP + FP); Rc = TP/(TP + FN); Ac = (TP + TN)/(TP + FP + TN + FN);
else 
    Pr = TP/(TP + FP); Rc = TP/(TP + FN); Ac = (TP + TN)/(TP + FP + TN + FN);
end 

Pr
Rc
Ac

%extragere proprietati pentru clasificare
%fractional sky cover - FSC = number of cloudy pixels/total number of pixels
FSC = cloudPixels/number_pixels;
  disp(['FSC:' num2str(FSC)]); 

se = strel('disk', 2);
img_dilated = imdilate(imagine_afisare, se);

% Scadem din imaginea dilatata imaginea segmentata pentru a obtine perimetrul
img_perimeter = imsubtract(img_dilated, imagine_afisare);

% Numaram pixelii ramasi pentru a obtine perimetrul
perimeter_length = sum(img_perimeter(:));

% Afisam imaginea cu perimetrul evidențiat
% subplot(1, 3, 3);
% imshow(img_perimeter);
% title('Conturul');

if cloudPixels == 0 
    CB = 0;
else
CB = perimeter_length/cloudPixels;
end
disp(['CB: ' num2str(CB)]);

TH_sum = 0;
index = 1;
R_values = [];
B_values = [];
for o = 1:m
    for p = 1:n
        if(imagine_afisare(o, p) == 1)  %verific daca pixelul în cauză este un pixel care aparține norilor
            
            TH_ratio = red_channel(o, p)/blue_channel(o, p);
            TH_sum = TH_sum + TH_ratio;
            %TH_count = TH_count + 1;
R_values(index) = red_channel(o, p);
            B_values(index) = blue_channel(o, p);
            index = index + 1;
        end
    end
end
%if TH_count == 0
if cloudPixels == 0
    TH = 0;
else
TH = TH_sum/cloudPixels;
end
disp(['TH:' num2str(TH)]);
disp(' ');

  % Stochează valorile în vectori
    FSC_values(w) = FSC;
    CB_values(w) = CB;
    TH_values(w) = TH;
std_values(w) = standardDeviation;
  % Concatenarea vectorilor intr-o matrice de caracteristici
% feature_matrix = horzcat(FSC_values, CB_values, TH_values, imageNames);

% Matricea de caracteristici pentru cele 32 de imagini (32x3)
imageFeatures = [FSC_values CB_values TH_values];

%alg. clasificare
labels = ["clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa1", "clasa2", "clasa2", "clasa2", "clasa2", "clasa2", "clasa2", "clasa2", "clasa2", "clasa2", "clasa3", "clasa3", "clasa3", "clasa4", "clasa4", "clasa4", "clasa4", "clasa4", "clasa4"];

distances = zeros(numImages, numImages);
predictedLabels = strings(numImages, 1);

for i = 1:numImages
    referenceFeatures = imageFeatures(i, :);
    minDistance = Inf;
    
    for j = 1:numImages
        if j == i
            continue;  % Evită comparația imaginii de referință cu ea însăși
        end
        
        currentFeatures = imageFeatures(j, :);
        
        differences = (referenceFeatures - currentFeatures).^2;
        sumSquaredDifferences = sum(differences);
        euclideanDistance = sqrt(sumSquaredDifferences);
        distances(i, j) = euclideanDistance;
        
        if euclideanDistance < minDistance
            minDistance = euclideanDistance;
            predictedLabels(i) = labels(j);
        end
    end
end

feature_matrix = horzcat(FSC_values, CB_values, TH_values, imageNames, transpose(labels), predictedLabels);

True_cl1 = 0;
False_cl1_cl2 = 0;
False_cl1_cl3 = 0;
False_cl1_cl4 = 0;
False_cl2_cl1 = 0;
True_cl2 = 0;
False_cl2_cl3 = 0;
False_cl2_cl4 = 0;
False_cl3_cl1 = 0;
False_cl3_cl2 = 0;
True_cl3 = 0;
False_cl3_cl4 = 0;
False_cl4_cl1 = 0;
False_cl4_cl2 = 0;
False_cl4_cl3 = 0;
True_cl4 = 0;


correct_cl1 = 0;
correct_cl2 = 0;
correct_cl3 = 0;
correct_cl4 = 0;

for q = 1:numImages
if (labels(q) == predictedLabels(q) && labels(q) == "clasa1" && predictedLabels(q) == "clasa1" )
    True_cl1 = True_cl1 + 1;
elseif (labels(q) ~= predictedLabels(q) && labels(q) == "clasa1" && predictedLabels(q) == "clasa2" )
    False_cl1_cl2 = False_cl1_cl2 + 1;
elseif (labels(q) ~= predictedLabels(q) && labels(q) == "clasa1" && predictedLabels(q) == "clasa3")
    False_cl1_cl3 = False_cl1_cl3 + 1;
elseif (labels(q) ~= predictedLabels(q) && labels(q) == "clasa1" && predictedLabels(q) == "clasa4")
    False_cl1_cl4 = False_cl1_cl4 + 1;
elseif( labels(q) ~= predictedLabels(q) && labels(q) == "clasa2" && predictedLabels(q) == "clasa1")
    False_cl2_cl1 = False_cl2_cl1 + 1;
elseif(labels(q) == predictedLabels(q) && labels(q) == "clasa2" && predictedLabels(q) == "clasa2")
    True_cl2 = True_cl2 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa2" && predictedLabels(q) == "clasa3")
    False_cl2_cl3 = False_cl2_cl3 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa2" && predictedLabels(q) == "clasa4")
    False_cl2_cl4 = False_cl2_cl4 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa3" && predictedLabels(q) == "clasa1")
    False_cl3_cl1 = False_cl3_cl1 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa3" && predictedLabels(q) == "clasa2")
    False_cl3_cl2 = False_cl3_cl2 + 1;
elseif(labels(q) == predictedLabels(q) && labels(q) == "clasa3" && predictedLabels(q) == "clasa3")
    True_cl3 = True_cl3 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa3" && predictedLabels(q) == "clasa4")
    False_cl3_cl4 = False_cl3_cl4 + 1;
elseif (labels(q) ~= predictedLabels(q) && labels(q) == "clasa4" && predictedLabels(q) == "clasa1")
    False_cl4_cl1 = False_cl4_cl1 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(q) == "clasa4" && predictedLabels(q) == "clasa2")
    False_cl4_cl2 = False_cl4_cl2 + 1;
elseif(labels(q) ~= predictedLabels(q) && labels(i) == "Clasa4" && predictedLabels(q) == "clasa3")
    False_cl4_cl3 = False_cl4_cl3 + 1;
else 
    True_cl4 = True_cl4 + 1;
end

correct_cl1 = sum(labels == "clasa1");%numarul de poze care apartin clasei 1
correct_cl2 = sum(labels == "clasa2");%numarul de poze care apartin clasei 2 
correct_cl3 = sum(labels == "clasa3");%numarul de poze care apartin clasei 3 
correct_cl4 = sum(labels == "clasa4");%numarul de poze care apartin clasei 4
end

confusionMatrix2 = [True_cl1 False_cl1_cl2 False_cl1_cl3 False_cl1_cl4; False_cl2_cl1 True_cl2 False_cl2_cl3 False_cl2_cl4;
    False_cl3_cl1 False_cl3_cl2 True_cl3 False_cl3_cl4; False_cl4_cl1 False_cl4_cl2 False_cl4_cl3 True_cl4];

%performante
correctClassification = (True_cl1 + True_cl2 + True_cl3 + True_cl4)/numImages;

correctClassification_class1 = True_cl1/correct_cl1;
correctClassification_class2 = True_cl2/correct_cl2;
correctClassification_class3 = True_cl3/correct_cl3;
correctClassification_class4 = True_cl4/correct_cl4;

%procentul imaginilor de clasa 1 clasificate ca si clasa 2
procent_cl1_cl2 = False_cl1_cl2/correct_cl1;
%procentul imaginilor din clasa 1 clasificate ca si clasa 3
procent_cl1_cl3 = False_cl1_cl3/correct_cl1;
%procentul imaginilor din clasa 1 clasficiate ca si clasa 4
procent_cl1_cl4 = False_cl1_cl4/correct_cl1;

%procentul imaginilor de clasa 2 clasficiate ca si clasa 1
procent_cl2_cl1 = False_cl2_cl1/correct_cl2;
%procentul imaginilor din clasa 2 clasificate ca si clasa 3
procent_cl2_cl3 = False_cl2_cl3/correct_cl2;
%procentul imaginilor din clasa 2 clasificate ca si clasa 4
procent_cl2_cl4 = False_cl2_cl4/correct_cl2;

%procentul imaginilor din clasa 3 clasificate ca si clasa 1
procent_cl3_cl1 = False_cl3_cl1/correct_cl3;
%procentul imaginilor din clasa 3 clasificate ca si clasa 2
procent_cl3_cl2 = False_cl3_cl2/correct_cl3;
%procentul imaginulor din clasa 3 clasificate ca si clasa 4
procent_cl3_cl4 = False_cl3_cl4/correct_cl3;

%procentul imaginilor din clasa 4 clasificate ca si clasa 1
procent_cl4_cl1 = False_cl4_cl1/correct_cl4;
%procentul imaginilor din clasa 4 clasificate ca si clasa 2
procent_cl4_cl2 = False_cl4_cl2/correct_cl4;
%procentul imaginilor din clasa 4 clasificate ca si clasa 3
procent_cl4_cl3 = False_cl4_cl3/correct_cl4;


end

