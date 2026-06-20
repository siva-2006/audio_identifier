import streamlit as st
import pandas as pd
import os
import pickle
import librosa
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
from collections import Counter
import matplotlib.pyplot as plt
import re

st.set_page_config(
    page_title="EE200 Project Demo",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .course-title {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #FFFFFF;
        letter-spacing: -0.05rem;
        margin-bottom: 0px;
    }
    
    .demo-subtitle {
        font-size: 1.2rem !important;
        color: #3182CE;
        font-weight: 600;
        margin-top: 0px;
        margin-bottom: 0.5rem;
    }
    
    .project-description {
        font-size: 0.95rem !important;
        color: #A0AEC0;
        margin-bottom: 2rem;
        line-height: 1.6;
        background-color: #1A202C;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2D3748;
    }
    
    .tab-info-text {
        font-size: 0.95rem;
        color: #A0AEC0;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    .step-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .step-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #00F0FF;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin-bottom: 0.2rem;
    }
    
    .step-title {
        margin: 0px 0px 0.5rem 0px;
        color: #FFFFFF;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .match-banner {
        background: linear-gradient(90deg, #1A365D 0%, #2A4365 100%);
        border-left: 5px solid #3182CE;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    .text-cyan { color: #00F0FF; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

def to_display_name(song_id):
    if not song_id or song_id.lower() == "none":
        return song_id
    return re.sub(r'(\b\w+)_([tsm]|re|ve|ll|d)\b', r"\1'\2", song_id)

def get_spectrogram(audio_data, fs, nperseg=1024, noverlap=512):
    f, t, Sxx = signal.spectrogram(audio_data, fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    return f, t, 10 * np.log10(Sxx + 1e-10)

def get_constellation(Sxx_db, neighborhood_size=20, threshold_percentile=90):
    data_max = ndimage.maximum_filter(Sxx_db, size=neighborhood_size)
    maxima = (Sxx_db == data_max)
    threshold = np.percentile(Sxx_db, threshold_percentile)
    peaks = maxima & (Sxx_db > threshold)
    freq_bins, time_frames = np.where(peaks)
    return list(zip(time_frames, freq_bins))

def generate_hashes(peaks, song_name, delay_min=1, delay_max=50, target_zone_size=5):
    peaks = sorted(peaks, key=lambda x: x[0])
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, target_zone_size + 1):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                if delay_min <= delta_t <= delay_max:
                    hashes.append(((f1, f2, delta_t), (song_name, t1)))
    return hashes

def match_query_clip(audio_data, fs, database):
    f, t, Sxx_db = get_spectrogram(audio_data, fs)
    peaks = get_constellation(Sxx_db)
    query_hashes = generate_hashes(peaks, song_name="query") 
    
    song_offsets = {}
    for hash_key, query_val in query_hashes:
        query_time = query_val[1]
        if hash_key in database:
            for db_song, db_time in database[hash_key]:
                offset = db_time - query_time
                if db_song not in song_offsets:
                    song_offsets[db_song] = []
                song_offsets[db_song].append(offset)
                
    best_song = "none"
    max_matches = 0
    best_histogram_data = []
    winning_offset_frames = 0
    candidate_scores = {}
    
    for song, offsets in song_offsets.items():
        if not offsets:
            continue
        offset_counts = Counter(offsets)
        most_common_offset, count = offset_counts.most_common(1)[0]
        candidate_scores[song] = count
        if count > max_matches:
            max_matches = count
            best_song = song
            best_histogram_data = offsets
            winning_offset_frames = most_common_offset
            
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return best_song, max_matches, best_histogram_data, f, t, Sxx_db, peaks, sorted_candidates, winning_offset_frames

def reconstruct_song_constellations(database):
    song_peaks = {}
    for hash_key, instances in database.items():
        f1, f2, delta_t = hash_key
        for song_name, t1 in instances:
            if song_name not in song_peaks:
                song_peaks[song_name] = set()
            song_peaks[song_name].add((t1, f1))
            song_peaks[song_name].add((t1 + delta_t, f2))
    return song_peaks

@st.cache_resource
def load_cached_database(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), True
    return None, False

database_path = "FINGERPRINTS/song_database.pkl"
database, db_loaded = load_cached_database(database_path)

st.markdown('<h1 class="course-title">EE200 Signals, Systems and Networks</h1>', unsafe_allow_html=True)
st.markdown('<p class="demo-subtitle">Project Demo | Robust Audio Fingerprinting System</p>', unsafe_allow_html=True)
st.markdown("""
    <div class="project-description">
        <strong>Project Overview:</strong> This system applies principles of time-frequency analysis to implement an audio identification pipeline. 
        Raw acoustic waveforms are converted into 2D short-time spectrogram matrices using a balanced 1024-sample window with a 50% overlap. 
        A localized maximum filter extracts high-entropy structural peaks to form a unique "constellation map". Nearby peaks are paired into 
        invariant hashes to robustly match queries against a database by analyzing coherent time alignment offset histograms.
    </div>
""", unsafe_allow_html=True)

window = st.tabs(["| Database Tracks Explorer", "| Live Upload & Identification", "| Automated Batch Mode"])

with window[0]:
    st.markdown("<h3 style='color:#FFF; font-weight:600; margin-bottom:0.2rem;'>Global Database Fingerprint Constellations</h3>", unsafe_allow_html=True)
    st.markdown("<p class='tab-info-text'>This window reconstructs and displays the complete spectral constellation anchor maps stored inside the active fingerprint index. It visualizes the distribution profile for all existing database tracks simultaneously.</p>", unsafe_allow_html=True)
    
    if not db_loaded:
        st.error("No database snapshot found. Ensure your database file is uploaded.")
    else:
        with st.spinner("Extracting tracking configurations..."):
            song_maps = reconstruct_song_constellations(database)
            
        tracks = list(song_maps.keys())
        if not tracks:
            st.info("The database contains no indexed keys.")
        else:
            cols_per_row = 3
            for i in range(0, len(tracks), cols_per_row):
                row_tracks = tracks[i:i + cols_per_row]
                columns = st.columns(len(row_tracks))
                
                for idx, track_name in enumerate(row_tracks):
                    with columns[idx]:
                        st.markdown(f"""
                            <div style='background-color:#1A202C; border:1px solid #2D3748; padding:0.5rem 1rem; border-radius:6px; margin-bottom:0.2rem;'>
                                <span style='font-size:0.9rem; font-weight:600; color:#00F0FF;'>{to_display_name(track_name)}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        coords = list(song_maps[track_name])
                        t_coords = [c[0] for c in coords]
                        f_coords = [c[1] for c in coords]
                        
                        fig_grid, ax_grid = plt.subplots(figsize=(5, 2.5), facecolor='#1A202C')
                        ax_grid.set_facecolor('#1A202C')
                        ax_grid.scatter(t_coords, f_coords, color='#00F0FF', s=1.5, marker='o', alpha=0.4)
                        ax_grid.set_ylim(0, 512)
                        
                        ax_grid.tick_params(colors='#718096', labelsize=7)
                        ax_grid.set_xlabel("Time (Frames)", fontsize=7, color='#718096')
                        ax_grid.set_ylabel("Freq Bin", fontsize=7, color='#718096')
                        for spine in ax_grid.spines.values():
                            spine.set_edgecolor('#2D3748')
                            
                        st.pyplot(fig_grid, facecolor='#1A202C')
                        plt.close(fig_grid)

with window[1]:
    st.markdown("<h3 style='color:#FFF; font-weight:600; margin-bottom:0.2rem;'>Search Query Terminal</h3>", unsafe_allow_html=True)
    st.markdown("<p class='tab-info-text'>Upload a single short unknown audio sample to analyze its distinct signal patterns. The pipeline isolates its time-frequency parameters, overlays its timeline window against the best match, and displays the alignment matrix histogram.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop clip file below (.mp3 / .wav)", type=["mp3", "wav"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if not db_loaded:
            st.error("System missing structural reference matrix indices.")
        else:
            with st.spinner("Processing architectural verification steps..."):
                data, fs = librosa.load(uploaded_file, sr=None)
                matched_song, score, hist_data, f, t, Sxx_db, peaks, candidates, win_offset_frames = match_query_clip(data, fs, database)
                
                if score <= 2:
                    matched_song = "none"
            
            st.markdown(f"""
                <div class="match-banner">
                    <span style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1rem; color:#90CDF4; font-weight:600;">Match Identified</span>
                    <h2 style="margin:0.2rem 0; color:#FFFFFF; font-size:2.2rem; font-weight:700;">{to_display_name(matched_song)}</h2>
                    <p style="margin:0; color:#E2E8F0; font-size:0.95rem;">System Confidence Score: <span class="text-cyan">{score} aligned hashes</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("System Candidate Rankings List"):
                if candidates:
                    df_cand = pd.DataFrame(candidates, columns=["Track Title", "Hash Intersection Spike"])
                    df_cand["Track Title"] = df_cand["Track Title"].apply(to_display_name)
                    st.dataframe(df_cand, use_container_width=True, hide_index=True)
            
            st.markdown("<h3 style='color:#FFF; font-weight:600; margin-top:2rem; margin-bottom:1rem;'>3-Step Diagnostic Pipeline</h3>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class="step-card">
                    <div class="step-header">Step 1</div>
                    <div class="step-title">Query Clip Feature Extraction</div>
                    <p style="font-size:0.85rem; color:#A0AEC0; margin:0;">Displays the computed spectrogram and extracted high-entropy maximum peaks (cyan landmarks) from the user's uploaded audio snippet.</p>
                </div>
            """, unsafe_allow_html=True)
            
            fig_step1, ax_step1 = plt.subplots(figsize=(12, 4), facecolor='#1A202C')
            ax_step1.set_facecolor('#1A202C')
            ax_step1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno')
            
            q_frames = [p[0] for p in peaks]
            q_bins = [p[1] for p in peaks]
            ax_step1.scatter(t[q_frames], f[q_bins], color='#00F0FF', s=10, marker='x', alpha=0.8)
            ax_step1.set_ylim(0, 5000)
            ax_step1.tick_params(colors='#A0AEC0', labelsize=8)
            ax_step1.set_xlabel("Time (Seconds)", color='#A0AEC0', fontsize=9)
            ax_step1.set_ylabel("Frequency (Hz)", color='#A0AEC0', fontsize=9)
            st.pyplot(fig_step1, facecolor='#1A202C')
            plt.close(fig_step1)
            
            st.markdown("""
                <div class="step-card">
                    <div class="step-header">Step 2</div>
                    <div class="step-title">Database Alignment Localization</div>
                    <p style="font-size:0.85rem; color:#A0AEC0; margin:0;">Displays the complete constellation profile of the matched song retrieved from the database. The <span class='text-cyan'>shaded blue window</span> accurately pinpoints where the query clip sits along the full timeline.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if matched_song != "none" and matched_song in song_maps:
                full_coords = list(song_maps[matched_song])
                full_t = [c[0] for c in full_coords]
                full_f = [c[1] for c in full_coords]
                
                fig_step2, ax_step2 = plt.subplots(figsize=(12, 4), facecolor='#1A202C')
                ax_step2.set_facecolor('#1A202C')
                ax_step2.scatter(full_t, full_f, color='#718096', s=2, marker='o', alpha=0.4, label="Full Song Anchor Points")
                
                clip_duration_frames = len(t)
                start_frame = win_offset_frames
                end_frame = start_frame + clip_duration_frames
                
                ax_step2.axvspan(start_frame, end_frame, color='#3182CE', alpha=0.4, label="Identified Query Window Location")
                ax_step2.set_ylim(0, 512)
                ax_step2.tick_params(colors='#A0AEC0', labelsize=8)
                ax_step2.set_xlabel("Full Song Timeline (Spectrogram Frames)", color='#A0AEC0', fontsize=9)
                ax_step2.set_ylabel("Frequency Bin Index", color='#A0AEC0', fontsize=9)
                ax_step2.legend(loc="upper right", framealpha=0.1, labelcolor='#FFF')
                st.pyplot(fig_step2, facecolor='#1A202C')
                plt.close(fig_step2)
            else:
                st.info("Time alignment synchronization maps are omitted for unmatched or unidentified signals.")
                
            st.markdown("""
                <div class="step-card">
                    <div class="step-header">Step 3</div>
                    <div class="step-title">Time Offset Histogram Decision</div>
                    <p style="font-size:0.85rem; color:#A0AEC0; margin:0;">Plots the distribution of the structural hash differences. A singular, high-magnitude spike verifies that the temporal relationships match a target reference track perfectly.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if matched_song != "none" and hist_data:
                frame_duration = t[1] - t[0] if len(t) > 1 else 0.023
                time_converted_offsets = [o * frame_duration for o in hist_data]
                
                fig_step3, ax_step3 = plt.subplots(figsize=(12, 4), facecolor='#1A202C')
                ax_step3.set_facecolor('#1A202C')
                ax_step3.hist(time_converted_offsets, bins=60, color='#3182CE', edgecolor='#1A202C', alpha=0.9)
                ax_step3.tick_params(colors='#A0AEC0', labelsize=8)
                ax_step3.set_xlabel("Time Offset Delta (Seconds)", color='#A0AEC0', fontsize=9)
                ax_step3.set_ylabel("Coincidental Tally Count", color='#A0AEC0', fontsize=9)
                ax_step3.grid(color='#2D3748', linestyle='--', linewidth=0.5)
                st.pyplot(fig_step3, facecolor='#1A202C')
                plt.close(fig_step3)
            else:
                st.info("Insufficient intersection scores to compute structural histograms.")

with window[2]:
    st.markdown("<h3 style='color:#FFF; font-weight:600; margin-bottom:0.2rem;'>Identify Many Clips at Once</h3>", unsafe_allow_html=True)
    st.markdown("<p class='tab-info-text'>Upload a set of query clips simultaneously. Each is identified against the currently indexed library, and the results are compiled into a standardized <code>results.csv</code> sheet containing the columns <strong>filename</strong> and <strong>prediction</strong>.</p>", unsafe_allow_html=True)
    
    batch_files = st.file_uploader(
        "Upload multiple query clips...", 
        type=["mp3", "wav"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if batch_files:
        if st.button("Run batch"):
            if not db_loaded:
                st.error("Active hash indices required to initiate batch evaluations.")
            else:
                results_list = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file_obj in enumerate(batch_files):
                    filename = uploaded_file_obj.name
                    status_text.caption(f"Analyzing track {idx+1}/{len(batch_files)}: `{filename}`")
                    
                    try:
                        data, fs = librosa.load(uploaded_file_obj, sr=None)
                        song_pred, score, _, _, _, _, _, _, _ = match_query_clip(data, fs, database)
                        
                        if score <= 2: 
                            song_pred = "none"
                        
                        results_list.append([filename, song_pred])
                    except Exception as e:
                        results_list.append([filename, "none"])
                        
                    progress_bar.progress((idx + 1) / len(batch_files))
                
                status_text.empty()
                
                df = pd.DataFrame(results_list, columns=['filename', 'prediction'])
                df.to_csv("results.csv", index=False)
                
                st.success("Batch identification execution complete.")
                
                df_display = df.copy()
                df_display["prediction"] = df_display["prediction"].apply(to_display_name)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results.csv Spreadsheet",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv"
                )