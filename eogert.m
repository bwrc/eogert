function eogert
% function eogert
% eogert - EOG Event Recognizer Tool
% A realtime probabilistic algorithm which detects fixations, saccades, and blinks from EOG signals.
% In this implementation, the signals are loaded from a file and read inside the "while true" loop; modify the code according to your needs in that part.

% $$$ Copyright (c) 2014 Brain Work Research Centre, Miika Toivanen <miika.toivanen@ttl.fi>
% $$$ 
% $$$ Permission is hereby granted, free of charge, to any person obtaining a copy
% $$$ of this software and associated documentation files (the "Software"), to deal
% $$$ in the Software without restriction, including without limitation the rights
% $$$ to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% $$$ copies of the Software, and to permit persons to whom the Software is
% $$$ furnished to do so, subject to the following conditions:
% $$$ 
% $$$ The above copyright notice and this permission notice shall be included in all
% $$$ copies or substantial portions of the Software.
% $$$ 
% $$$ THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% $$$ IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% $$$ FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% $$$ AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% $$$ LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% $$$ OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% $$$ SOFTWARE.

load test001.mat EOGh EOGv fs; % Load the EOG signals from a file (in reality, they would be inputted in the loop), available in the same folder
 
BAR_PLOT=0;     % bar plot the probabilities realtime?
PLOT1D=0;       % plot the likelihoods and feature values realtime?
PLOTEM = [0 0]; % plot the EM-GMM iterations? (Separately for both features)

%% Set the duration thresholds
MIN_SACCADE_LEN = 0.010;  % the minimum saccade length in secs
MAX_SACCADE_LEN = 0.200;  % the maximum saccade length (i.e. the length of the Dn feature buffer)
MIN_BLINK_LEN = 0.030;    % the minimum blink length
MAX_BLINK_LEN = 0.500;    % the maximum blink length (i.e. the length of the Dv feature buffer)
MIN_SACCADE_GAP = 0.100;  % the default minimum distance between two saccades
TRAIN_SECS = 60;          % the duration of training period in seconds

%% Design filters
FIRlen = 150;
pass_limit1 = 1;
pass_limit2 = 40;
Bfir1 = fir1(FIRlen, pass_limit1/(fs/2));     % the FIR filter for detecting the events
groupDelay = (FIRlen-1)/2;                    % group delay, i.e. linear phase of the FIR filter, is (M-1)/2 where M is the length of the filter
Bfir2 = fir1(FIRlen, pass_limit2/(fs/2));     % the FIR filter for computing accurate temporal values
hz1 = zeros(1,length(Bfir1)-1); vz1 = zeros(1,length(Bfir1)-1);
hz2 = zeros(1,length(Bfir2)-1); vz2 = zeros(1,length(Bfir2)-1);

%% Set some variables
saccade_on = 0;
saccade_samples = 0;
saccade_prob = 0;
blink_on = 0;
blink_samples = 0;
blink_prob = 0;
Nsacs = 0;
Nblinks = 0;
fix_prob_mass_between_saccades = 0;       % the probability mass of fixations between saccades
training_period = round(fs * TRAIN_SECS); % training period in samples

hf1_prev = 0; vf1_prev = 0; hf2_prev = 0; vf2_prev = 0;
hf1_train = []; vf1_train = [];
buffer_nd2 = zeros(1, round(MAX_SACCADE_LEN * fs)); % The buffer of Dn features
buffer_vf2 = zeros(1, round(MAX_BLINK_LEN * fs));   % The buffer of Dv features
buffer_psn = buffer_nd2;

n = 1;       % sample number
while true   % the eternity loop
   
   if PLOT1D | BAR_PLOT, set(gcf,'currentcharacter','a'), end
   if ~floor(mod(n, fs)); fprintf('%d  ', round(n/fs)); end
   
   %% Receive EOG samples
   % -- Here the samples are vector elements; in reality, they would be inputted here by socket etc. Modify according to your needs.
   h = EOGh(n);  % a sample of horizontal signal
   v = EOGv(n);  % a sample of vertical signal
   

   %% Realtime filtering
   [hf1 hz1] = filter(Bfir1,1,h,hz1);
   [vf1 vz1] = filter(Bfir1,1,v,vz1);
   [hf2 hz2] = filter(Bfir2,1,h,hz2);
   [vf2 vz2] = filter(Bfir2,1,v,vz2);

   
   %% Save the training samples
   if n<training_period
      hf1_train = [hf1_train hf1];
      vf1_train = [vf1_train vf1];
   end

   
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %% Detection stage (training period is now over)
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if n>training_period

      %% compute the derivates
      Hd1 = hf1 - hf1_prev;
      Vd1 = vf1 - vf1_prev;
      Hd2 = hf2 - hf2_prev;
      Vd2 = vf2 - vf2_prev;
      hf1_prev = hf1; vf1_prev = vf1;
      hf2_prev = hf2; vf2_prev = vf2;

      %% Find the max-min values of dv peaks (--> the "Dv" feature)
      if n==training_period+1
         curr_vd_max = Vd1;  % current maximum of Vd
         vd_prev = Vd1;
      end
      if Vd1 > vd_prev  % going up
         curr_vd_max = Vd1; peak_n = n;
      end
      vd_prev = Vd1; 
      dmm = curr_vd_max - Vd1 - abs(curr_vd_max+Vd1); % the "Dv" feature

      %% Compute the norms (--> the "Dn" feature)
      norm_D1 = norm([Hd1 Vd1]);
      norm_D2 = norm([Hd2 Vd2]);
      
      %% Update the feature buffers
      buffer_nd2(1:end-1) = buffer_nd2(2:end);
      buffer_nd2(end) = norm_D2;
      buffer_vf2(1:end-1) = buffer_vf2(2:end);
      buffer_vf2(end) = vf2;
            
            
      %% Compute likelihoods using asymmetric distributions.
      if norm_D1 > mu_fix
         Lf = normpdf(norm_D1, mu_fix, sigma_fix);     % likelihood of the fixation point
      else
         Lf = normpdf(mu_fix, mu_fix, sigma_fix);      % use the maximum lkh value
      end
      if norm_D1 < mu_bs
         Lbs = normpdf(norm_D1, mu_bs, sigma_bs);      % likelihood of the blink or saccade
      else
         Lbs = normpdf(mu_bs, mu_bs, sigma_bs);
      end
      if dmm > mu_sac
         Ls = normpdf(dmm, mu_sac, sigma_sac);         % likelihood (dmm) of the saccade
      else
         Ls = normpdf(mu_sac, mu_sac, sigma_sac);         
      end
      if dmm < mu_bli
         Lb = normpdf(dmm, mu_bli, sigma_bli);         % likelihood (dmm) of the blink
      else
         Lb = normpdf(mu_bli, mu_bli, sigma_bli);
      end

      %% Compute the posterior probabilities
      evi_norm = Lf*prior_fix + Lbs*prior_bs;      % evidence of norm data, i.e. the normalization constant
      pfn = Lf * prior_fix / evi_norm;             % normalized probability for the sample to be a fixation point
      psbn = 1 - pfn;                              % normalized probability for the sample to be a saccade or blink
      evi_dmm = Ls*prior_sac + Lb*prior_bli;       % evidence (dmm), i.e. the normalization constant         
      pbn = psbn * Lb * prior_bli / evi_dmm;       % normalized probability for the sample to be a blink
      psn = psbn * Ls * prior_sac / evi_dmm;       % normalized probability for the sample to be a saccade

      %% Update the probability buffers
      buffer_psn(1:end-1) = buffer_psn(2:end);
      buffer_psn(end) = psn;

      fix_prob_mass_between_saccades = fix_prob_mass_between_saccades + pfn; % increase the probability mass of fixations between saccades
      
      %% Detect the saccades using the probabilities and compute the duration and peak probabilities of them
      if all(psn > [pfn pbn])  % saccade is now happening
         if saccade_samples==0, sac_on_start_n = n; end
         saccade_on = 1;
         saccade_samples = saccade_samples+1;
         saccade_prob = saccade_prob + psn;    % an estimated probability mass of this saccade
      elseif saccade_on        % saccade just ended
         % We must have certain probability mass of saccade samples in order to accept this saccade:
         if saccade_prob > MIN_SACCADE_LEN * fs

            % -- Compute the saccade duration and starting time
            [buffer_peak peak_in_buffer] = max(buffer_nd2);
            if peak_in_buffer==1, peak_start_in_buffer = 1; else
               for peak_start_in_buffer=peak_in_buffer-1:-1:1;
                  if buffer_nd2(peak_start_in_buffer)-buffer_nd2(peak_start_in_buffer+1)>0, break, end
               end
               peak_start_in_buffer = peak_start_in_buffer+1;
            end
            if peak_in_buffer==length(buffer_nd2), peak_end_in_buffer = length(buffer_nd2); else  % !!
               for peak_end_in_buffer=peak_in_buffer+1:length(buffer_nd2); 
                  if buffer_nd2(peak_end_in_buffer)-buffer_nd2(peak_end_in_buffer-1)>0, break, end 
               end
            end
            peak_end_in_buffer = peak_end_in_buffer-1;
            saccade_dur = peak_end_in_buffer - peak_start_in_buffer;
            saccade_prob_mass = sum( buffer_psn(round(peak_start_in_buffer) : end)); % the "real" probability mass of this saccade
            buffer_start = round(max(1, n - length(buffer_nd2)));
            saccade_start_n = buffer_start + peak_start_in_buffer - groupDelay - 1;
                        
            % -- Final checks if this saccade was ok
            saccade_ok = 1;
            if saccade_prob_mass < MIN_SACCADE_LEN * fs, saccade_ok = 0; end
            if Nblinks>0  % check if the previous blink ends after start of this saccade --> this saccade is then "ending" of a blink and not real saccade
               if saccade_start_n/fs < BLI_START(end) + BLI_DUR(end); saccade_ok = 0; end
            end
            if Nsacs>0
                 % The time between previous saccade and this and the fixation probability mass there must exceed the minimum saccade gap
               if fix_prob_mass_between_saccades < MIN_SACCADE_GAP * fs, saccade_ok = 0; end
               if saccade_start_n/fs - (SAC_START(end)+SAC_DUR(end)) < MIN_SACCADE_GAP, saccade_ok = 0; end
            end

            if saccade_ok               
               % -- This is a real saccade -->  save the saccade parameter into vectors
               Nsacs = Nsacs+1;
               SAC_START(Nsacs) = saccade_start_n/fs;                   % an estimated start time of this saccade
               SAC_DUR(Nsacs) = saccade_dur/fs;                         % the saccade duration
               SAC_PROB(Nsacs) = saccade_prob/saccade_samples;          % the average saccade probability
               fix_prob_mass_between_saccades = 0;
               sac_on_end_prev_n = n;
            end
         end
         saccade_on = 0;
         saccade_samples = 0;
         saccade_prob = 0;         

      end  % --- End of SACCADE blobk

         
      
      %% Detect the blinks using the probabilities and compute the duration and peak probabilities of them
      if all(pbn > [pfn psn])  % blink is now happening
         blink_on = 1;
         blink_samples = blink_samples+1;
         blink_prob = blink_prob + pbn;
      elseif blink_on        % blink just ended
         % We must have enough blink probability mass to conclude it was a blink (blink_prob contains only 1/4 of the blink):
         if blink_prob > MIN_BLINK_LEN/4 * fs
            
            % -- Compute the blink duration and starting times
            buffer_dvf2 = diff(buffer_vf2);
            [buffer_peak_dvf2   peak_in_buffer_dvf2] = max(buffer_dvf2);
            for peak_start_in_buffer_dvf2=peak_in_buffer_dvf2:-1:1, if buffer_dvf2(peak_start_in_buffer_dvf2)<0.1*buffer_peak_dvf2, break, end, end
            [buffer_peak_vf2 peak_in_buffer_vf2] = max(buffer_vf2);
            blink_dur = 2 * (peak_in_buffer_vf2 - peak_start_in_buffer_dvf2);   % an estimated blink duration (in samples) assuming symmetric peak

            if blink_dur > MIN_BLINK_LEN * fs  % a final check if this blink was ok               
               % -- This is a real blink --> save the blink parameters into a vector
               Nblinks = Nblinks+1;
               buffer_start = round(max(1, n - MAX_BLINK_LEN * fs));
               BLI_START(Nblinks) = (buffer_start + peak_start_in_buffer_dvf2 - groupDelay)/fs;   % an estimated start time of this blink
               BLI_DUR(Nblinks) = blink_dur/fs;                                                   % the blink durations
               BLI_PROB(Nblinks) = blink_prob/blink_samples;                                      % the average blink probability
            end

            % -- Blinks form a sac-blink-sac sequence so remove the most recent saccade (the first "sac" in the sequence) if they overlap as it was not "real" saccade
            if Nsacs>0
               if BLI_START(end) < SAC_START(end)+SAC_DUR(end)
                  SAC_START = SAC_START(1:end-1); SAC_DUR = SAC_DUR(1:end-1); SAC_PROB = SAC_PROB(1:end-1); Nsacs = Nsacs-1;
               end
            end
         end
         
         blink_on = 0;
         blink_samples = 0;
         blink_prob = 0;
         
      end  % --- end of BLINK block
      

      % save the results (occasionally) into tmp.mat
      if ~mod(n,100) & Nsacs>0 & Nblinks>0; save tmp.mat BLI_* SAC_*; end

      % PLOTTING
      Nskip = 3; if saccade_on | blink_on, Nskip = 1; end     % slow down the plotting when saccade or blink is on
      if BAR_PLOT & ~mod(n,Nskip)
         bar([pfn psn pbn]); set(gca,'xticklabel',{'P(fixation)' ; 'P(saccade)' ; 'P(blink)'}); set(gca,'ylim',[0 1]); title(sprintf('t = %.2f', n/fs)), drawnow
      end
      if PLOT1D & ~mod(n,Nskip)
         col = 'b'; ms = 10; if saccade_on, col = 'g'; ms = 20; elseif blink_on, col = 'r'; ms = 20; end
         plot(x, normpdf(x,mu_fix,sigma_fix), x, normpdf(x,mu_bs,sigma_bs)); hold on, set(gca,'ylim',[0 normpdf(mu_bs,mu_bs,sigma_bs)*10]);
         plot(x, normpdf(x,mu_sac,sigma_sac), 'g--', x,normpdf(x,mu_bli,sigma_bli) ,'r--');
         plot([0 norm_D1], [1 1]*max(get(gca,'ylim'))/3, 'linewidth', 20, 'color', col);
         plot([0 dmm], [1 1]*max(get(gca,'ylim'))/2, '--', 'linewidth', 15, 'color', col);
         title(sprintf('t = %.2f', n/fs)), ax=axis; set(gca,'xlim',[0 ax(2)]), hold off, drawnow
      end
      
      if PLOT1D | BAR_PLOT, if get(gcf,'currentcharacter')=='q'; return; end, end
      
   end  % -- end of the detection stage

   
   
   
   %%%%%%%%%%%%%%%%%%%%%
   %% The training stage
   %%%%%%%%%%%%%%%%%%%%%
      
   if n==training_period
      fprintf('\n');

      burn_off = 2*FIRlen - 1;
      dh_tr = diff(hf1_train(burn_off:end));
      dv_tr = diff(vf1_train(burn_off:end));      

      %% Find the norm peaks
      for j=1:length(dh_tr), norm_tr(j) = norm([dh_tr(j) dv_tr(j)]); end
      curr_max = norm_tr(1);
      curr_min = norm_tr(1);
      norm_peak = [];
      tmp_i = 1;
      for i=1:length(dv_tr)
         if norm_tr(i) > curr_max, curr_max = norm_tr(i); curr_min = curr_max; tmp_i = i; end
         if norm_tr(i) < curr_min, curr_min = norm_tr(i);
         else
            if curr_max>curr_min, norm_peak = [norm_peak curr_max]; end   % minimum just reached
            curr_max = curr_min;
         end
      end

      % --- In the future test data, use norm values to separate fixations from other events
      % let EM find the likelihood parameters
      [mu_norm, sigma_norm, P_norm] = EMgauss1D(sort(norm_peak), [1 length(norm_peak)], PLOTEM(1)); if PLOTEM(1), pause(1), end
      mu_fix = mu_norm(1); sigma_fix = sigma_norm(1); prior_fix = P_norm(1);
      mu_bs = mu_norm(2);  sigma_bs = sigma_norm(2);  prior_bs = P_norm(2);         


      %% Find max-min values of dv peaks
      curr_max = dv_tr(1);
      curr_min = dv_tr(1);
      diff_max_min = [];
      time_diff = []; CURR_MIN = []; CURR_MAX = []; NTR = [];
      tmp_i = 1;
      for i=1:length(dv_tr)
         if dv_tr(i) > curr_max, curr_max = dv_tr(i); curr_min = curr_max; tmp_i = i; end  % going up
         if dv_tr(i) < curr_min  % going down
            curr_min = dv_tr(i);
         else
            if curr_max>curr_min     % minimum just reached
               ntr = norm_tr(tmp_i); % probability of this peak corresponding to blink or saccade:
               p_bs = normpdf(ntr, mu_bs, sigma_bs) * prior_bs / (normpdf(ntr, mu_bs, sigma_bs) * prior_bs + normpdf(ntr, mu_fix, sigma_fix) * prior_fix);
               if p_bs > 2/3         % the peak in dv_tr probably stems from blink or saccade (let's collect just those)
                  feature = curr_max - curr_min - abs(curr_max+curr_min); % feature favors peaks that are symmetric about zero
                  if feature>0
                     diff_max_min = [diff_max_min feature];
                     time_diff = [time_diff i];
                     CURR_MAX = [CURR_MAX curr_max]; CURR_MIN = [CURR_MIN curr_min]; NTR = [NTR ntr];
                  end
               end
            end
            curr_max = curr_min;
         end
      end

      % --- In the future test data, use diff-max-min (dmm) values to separate blinks from saccades
      % let EM find the likelihood parameters
      [mu_dmm sigma_dmm P_mm] = EMgauss1D(sort(diff_max_min), [1 length(diff_max_min)], PLOTEM(2)); if PLOTEM(2), pause(1), end
      mu_sac = mu_dmm(1); sigma_sac = sigma_dmm(1); prior_sac = P_mm(1);
      mu_bli = mu_dmm(2); sigma_bli = sigma_dmm(2); prior_bli = P_mm(2);
      
      if PLOT1D, x = linspace(0, max([max(norm_peak) max(diff_max_min)])*1.1, 1000); end

      %save train_params.mat mu_* sigma_*;  % save the likelihood parameters (optional)

      clear hf1_train vf1_train dh_tr dv_tr norm_tr norm_peak diff_max_min          % clear the unnecessary training data
   end  % --- end of the training period

   n = n+1;
end   % --- end of the eternity loop

