function eogert_offline(h, v, fs, train_secs, MIN_SACCADE_GAP)
% function eogert_offline(h, v, fs, train_secs, MIN_SACCADE_GAP)
% A probabilistic realtime detector of EOG events. In the beginning, there is an unsupervised calibration phase whose duration is train_secs.
% In this implementation, the whole signals are given as input.
% INPUT: h = EOG horizontal signal, v = EOG vertical signal, fs = sampling frequency, train_secs =  number of seconds for training in the beginning
% optional input: MIN_SACCADE_GAP = the minimum gap between two saccades (otherwise set to the default value, 100 ms)
% OUTPUT: all the things that is computed (starts and durations of events etc.) is saved into tmpResults.mat
% typical usage: eogert_offline(EOGh, EOGv, fs, 60);

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

BAR_PLOT=0;     % bar plot the probabilities realtime?
PLOT1D=0;       % plot the likelihoods and feature values realtime?
PLOTEM = [0 0]; % plot the EM-GMM iterations? (Separately for both features)

%% Set the duration thresholds
if nargin<5
   MIN_SACCADE_GAP = 0.100;  % the default minimum distance between two saccades
end
MIN_SACCADE_LEN = 0.010;  % the minimum saccade length in secs
MAX_SACCADE_LEN = 0.150;  % the maximum saccade length (i.e. the length of the buffer) (in reality, not this long saccades can be detected!)
MIN_BLINK_LEN = 0.030;    % the minimum blink length
MAX_BLINK_LEN = 0.500;    % the maximum blink length (i.e. the length of the buffer)

%% Design filters
FIRlen = 150;    
pass_limit1 = 1;
pass_limit2 = 40;
[b,a] = butter(6, pass_limit1/(fs/2)); % IIR filter (will result in non-linear phase when filtered realtime!)
Bfir = fir1(FIRlen, pass_limit1/(fs/2));     % FIR filter (must be of high order in order to have the same performance of IIR with much smaller order)   
groupDelay = (FIRlen-1)/2;   % group delay, i.e. linear phase of the FIR filter, is (M-1)/2 where M is the length of the filter
Bfir2 = fir1(FIRlen, pass_limit2/(fs/2));
hz = zeros(1,length(Bfir)-1); vz = zeros(1,length(Bfir)-1); hz2 = zeros(1,length(Bfir)-1); vz2 = zeros(1,length(Bfir)-1);

%% Set some variables
saccade_on = 0;
saccade_samples = 0;
saccade_prob = 0;
blink_on = 0;
blink_samples = 0;
blink_prob = 0;
blink_peak_prob = 0;
Nsacs = 0;
Nblinks = 0;
training_period = round(fs * train_secs); % training period in samples

for n=1:length(h);
   set(gcf,'currentcharacter','a')
   if ~floor(mod(n, fs)); fprintf('%d  ', round(n/fs)); end

   %% realtime filtering
   [hf_rec(n) hz] = filter(Bfir,1,h(n),hz);
   [vf_rec(n) vz] = filter(Bfir,1,v(n),vz);
   [hf_rec2(n) hz2] = filter(Bfir2,1,h(n),hz2);
   [vf_rec2(n) vz2] = filter(Bfir2,1,v(n),vz2);

   if n>1
      Hd = hf_rec(n)-hf_rec(n-1);
      Vd = vf_rec(n)-vf_rec(n-1);
      Hd2 = hf_rec2(n)-hf_rec2(n-1);
      Vd2 = vf_rec2(n)-vf_rec2(n-1);
   end
   
   %% Detection stage (training period is now over)
   if n>training_period
      
      %% Find the max-min values of dv peaks
      if n==training_period+1
         curr_vd_max = Vd;  % current maximum of Vd
         vd_prev = Vd;
      end
      if Vd > vd_prev  % going up
         curr_vd_max = Vd; peak_n = n;
      end
      vd_prev = Vd; 
      dmm = curr_vd_max - Vd - abs(curr_vd_max+Vd); % dmm favors peaks that are symmetric about zero
      
      DMM(n) = dmm;      
      norm_D = norm([Hd Vd]); ND(n) = norm_D;
      norm_D2 = norm([Hd2 Vd2]); ND2(n) = norm_D2;

      %% Compute likelihoods. Let's assume asymmetric distributions.
      if norm_D > mu_fix
         Lf = normpdf(norm_D, mu_fix, sigma_fix);     % likelihood of the fixation point
      else
         Lf = normpdf(mu_fix, mu_fix, sigma_fix);     % use the maximum lkh value
      end
      if norm_D < mu_bs
         Lbs = normpdf(norm_D, mu_bs, sigma_bs);      % likelihood of the blink or saccade
      else
         Lbs = normpdf(mu_bs, mu_bs, sigma_bs);
      end
      if dmm > mu_sac
         Ls = normpdf(dmm, mu_sac, sigma_sac);        % likelihood (dmm) of the saccade
      else
         Ls = normpdf(mu_sac, mu_sac, sigma_sac);         
      end
      if dmm < mu_bli
         Lb = normpdf(dmm, mu_bli, sigma_bli);        % likelihood (dmm) of the blink
      else
         Lb = normpdf(mu_bli, mu_bli, sigma_bli);
      end
               
      evi_norm = Lf*prior_fix + Lbs*prior_bs;      % evidence of norm data, i.e. the normalization constant
      pfn = Lf * prior_fix / evi_norm;             % normalized probability for the sample to be a fixation point
      psbn = 1 - pfn;                              % normalized probability for the sample to be a saccade or blink

      evi_dmm = Ls*prior_sac + Lb*prior_bli;       % evidence (dmm), i.e. the normalization constant         
      pbn = psbn * Lb * prior_bli / evi_dmm;       % normalized probability for the sample to be a blink
      psn = psbn * Ls * prior_sac / evi_dmm;       % normalized probability for the sample to be a saccade

      PSN(n) = psn; PFN(n) = pfn;  % collect the saccade and fixation probabilities
      
      
      %% Detect the saccades using the probabilities and compute the duration and peak probabilities of them
      if all(psn > [pfn pbn])  % saccade is now happening (or was groupdDelay samples ago...)
         if saccade_samples==0, sac_on_start_n = n; end
         saccade_on = 1;
         saccade_samples = saccade_samples+1;
         saccade_prob = saccade_prob + psn; % an estimated probability mass of this saccade
      elseif saccade_on        % saccade just ended                              
         % We must have certain probability mass of saccade samples in order to accept this saccade:
         if saccade_prob > MIN_SACCADE_LEN * fs
            %% collect diff of samples to a buffer to compute the saccade duration and starting times
            buffer_start = round(max(1, n - MAX_SACCADE_LEN * fs));
            if Nsacs>0, buffer_start = round(max(buffer_start, (SAC_START(end) + SAC_DUR(end))*fs + groupDelay + 1)); end  % don't let them overlap
            buffer = ND2(buffer_start : end);
            [buffer_peak peak_in_buffer] = max(buffer);
            if peak_in_buffer==1, peak_start_in_buffer = 1; else
               for peak_start_in_buffer=peak_in_buffer-1:-1:1; if buffer(peak_start_in_buffer)-buffer(peak_start_in_buffer+1)>0, break, end, end
               peak_start_in_buffer = peak_start_in_buffer+1;
            end
            if peak_in_buffer==length(buffer), peak_end_in_buffer = length(buffer); else
               for peak_end_in_buffer=peak_in_buffer+1:length(buffer); if buffer(peak_end_in_buffer)-buffer(peak_end_in_buffer-1)>0, break, end, end               
            end
            peak_end_in_buffer = peak_end_in_buffer-1;
            saccade_dur = peak_end_in_buffer - peak_start_in_buffer;   % I think the "+ 1" in the end is basically useless
            
            saccade_prob_mass = sum( PSN(round(buffer_start + peak_start_in_buffer - 1) : end));
            saccade_start_n = buffer_start + peak_start_in_buffer - groupDelay - 1;
                        
            % final checks if this saccade was ok
            saccade_ok = 1;
            if saccade_prob_mass < MIN_SACCADE_LEN * fs, saccade_ok = 0; end
            if Nblinks>0  % check if the previous blink ends after start of this saccade --> this is "ending" of a blink
               if saccade_start_n/fs < BLI_START(end) + BLI_DUR(end); saccade_ok = 0; end
            end
            if Nsacs>0
               if sum(PFN(sac_on_end_prev_n : sac_on_start_n)) < MIN_SACCADE_GAP * fs, saccade_ok = 0; end  % The fixation probability mass between previous saccade and this
               if (saccade_start_n)/fs - (SAC_START(end)+SAC_DUR(end) ) < MIN_SACCADE_GAP, saccade_ok = 0; end
            end

            if saccade_ok
               Nsacs = Nsacs+1;
               SAC_START(Nsacs) = saccade_start_n/fs;                   % an estimated start time of saccade
               SAC_DUR(Nsacs) = saccade_dur/fs;                         % collect the saccade durations
               SAC_PROB(Nsacs) = saccade_prob/saccade_samples;          % collect the average saccade probabilities
               sac_on_end_prev_n = n;               
            end

         end
         saccade_on = 0;
         saccade_samples = 0;
         saccade_prob = 0;         
      
      end  %% end of SACCADE block

         
      
      %% Detect the blinks using the probabilities and compute the duration and peak probabilities of them
      if all(pbn > [pfn psn])  % blink is now happening (or was few samples ago as there is the group delay...)
         blink_on = 1;
         blink_samples = blink_samples+1;
         blink_prob = blink_prob + pbn;
      elseif blink_on        % blink just ended
         this_blink_peak = (peak_n - groupDelay)/fs;      % the blink peak time (subtract FIR's group delay)
         blink_ok = 1;
         if blink_prob < MIN_BLINK_LEN/4 * fs, blink_ok = 0; end   % we must have enough blink probability mass to conclude it was a blink (blink_prob contains only 1/4 of the blink)
         
         if blink_ok
            %% collect samples to a buffer to compute the blink duration and starting times
            buffer_start = round(max(1, n - MAX_BLINK_LEN * fs));
            if Nblinks>0, buffer_start = round(max(buffer_start, (BLI_START(end) + BLI_DUR(end))*fs + groupDelay + 1)); end % don't let them overlap
            buffer = diff(vf_rec2(buffer_start : end));
            buffer2 = vf_rec2(buffer_start : end);
            [buffer_peak   peak_in_buffer] = max(buffer);
            [buffer_peak2 peak_in_buffer2] = max(buffer2);
            for peak_start_in_buffer=peak_in_buffer:-1:1, if buffer(peak_start_in_buffer)<0.1*buffer_peak, break, end, end
            blink_dur = 2 * (peak_in_buffer2 - peak_start_in_buffer + 0.5);   % an estimated blink duration (in samples) assuming symmetric peak
            if blink_dur > MIN_BLINK_LEN * fs  % a final check if this blink was ok
               Nblinks = Nblinks+1;
               BLI_PEAK(Nblinks) = this_blink_peak;    % (a rough estimate, shouldn't be used)
               BLI_START(Nblinks) = (buffer_start + peak_start_in_buffer - groupDelay)/fs;   % a better estimate
               BLI_DUR(Nblinks) = blink_dur/fs;        % collect the blink durations
               BLI_PROB(Nblinks) = blink_prob/blink_samples;    % collect the average blink probabilities
               BLI_DMM(Nblinks) = max(DMM(end-10:end));               
            end
                        
            % blinks form a sac-blink-sac sequence so remove the most recent saccade (the first "sac" in the sequence) as it was not "real" saccade
            if Nsacs>0
               if BLI_START(end) < SAC_START(end)+SAC_DUR(end)
                  SAC_START = SAC_START(1:end-1); SAC_DUR = SAC_DUR(1:end-1); SAC_PROB = SAC_PROB(1:end-1); Nsacs = Nsacs-1;
               end
            end
         end
         
         blink_on = 0;
         blink_samples = 0;
         blink_prob = 0;
         blink_peak_prob = 0;         
         
      end      % end of BLINK block
      

      % save the results into tmpResults.mat (after every 100th sample)
      if ~mod(n,100) & Nsacs>0 & Nblinks>0; save tmpResults.mat BLI_* SAC_*; end

      % PLOTTING
      Nskip = 3; if saccade_on | blink_on, Nskip = 1; end            
      if BAR_PLOT & ~mod(n,Nskip), bar([pfn psn pbn]); set(gca,'xticklabel',{'P(fixation)' ; 'P(saccade)' ; 'P(blink)'}); set(gca,'ylim',[0 1]); title(sprintf('t = %.2f', n/fs)), drawnow, end
      if PLOT1D & ~mod(n,Nskip)
         col = 'b'; ms = 10; if saccade_on, col = 'g'; ms = 20; elseif blink_on, col = 'r'; ms = 20; end
         plot(x, normpdf(x,mu_fix,sigma_fix), x, normpdf(x,mu_bs,sigma_bs)); hold on, set(gca,'ylim',[0 normpdf(mu_bs,mu_bs,sigma_bs)*10]);
         plot(x, normpdf(x,mu_sac,sigma_sac), 'g--', x,normpdf(x,mu_bli,sigma_bli) ,'r--');
         plot([0 norm_D], [1 1]*max(get(gca,'ylim'))/3, 'linewidth', 20, 'color', col);
         plot([0 dmm], [1 1]*max(get(gca,'ylim'))/2, '--', 'linewidth', 15, 'color', col);
         title(sprintf('t = %.2f', n/fs)), ax=axis; set(gca,'xlim',[0 ax(2)]), hold off, drawnow
      end
      
      if get(gcf,'currentcharacter')=='q'; return; end
   end

   
   
   
   
   %% Use the first train_secs seconds for calibration / training
   if n==training_period
      fprintf('\n');

      burn_off = 2*FIRlen - 1;
      dh_tr = diff(hf_rec(burn_off:end)); dv_tr = diff(vf_rec(burn_off:end));      

      %% Find the norm peaks
      for j=1:length(dh_tr), norm_tr(j) = norm([dh_tr(j) dv_tr(j)]); end
      curr_max = norm_tr(1);
      curr_min = norm_tr(1);
      norm_peak = [];
      time_norm = [];
      tmp_i = 1;
      for i=1:length(dv_tr)
         if norm_tr(i) > curr_max, curr_max = norm_tr(i); curr_min = curr_max; tmp_i = i; end
         if norm_tr(i) < curr_min, curr_min = norm_tr(i);
         else
            if curr_max>curr_min, norm_peak = [norm_peak curr_max]; time_norm = [time_norm tmp_i]; end   % minimum just reached
            curr_max = curr_min;
         end
      end

      % --- In the future test data, use norm values to separate fixations from other events
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
      % let EM find the distribution parameters
      [mu_dmm sigma_dmm P_mm] = EMgauss1D(sort(diff_max_min), [1 length(diff_max_min)], PLOTEM(2)); if PLOTEM(2), pause(1), end
      mu_sac = mu_dmm(1); sigma_sac = sigma_dmm(1); prior_sac = P_mm(1);
      mu_bli = mu_dmm(2); sigma_bli = sigma_dmm(2); prior_bli = P_mm(2);
      
      if PLOT1D, x = linspace(0, max([max(norm_peak) max(diff_max_min)])*1.1, 1000); end
      %save train_params.mat mu_* sigma_*;  % save the likelihood parameters (optional)
      
   end  % end of the training period
   
end

