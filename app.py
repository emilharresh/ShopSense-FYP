import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ShopSense | Official Store",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME & STYLES ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

SHOP_ORANGE = "#ee4d2d" 
SHOP_RED = "#d0011b"

themes = {
    "light": {
        "bg_color": "#f5f5f5", 
        "card_bg": "#ffffff",
        "text_color": "#2c3e50",
        "price_color": SHOP_ORANGE,
        "shadow": "0 1px 2px 0 rgba(0,0,0,.1)"
    }
}

current_theme = themes['light']

st.markdown(f"""
<style>
    /* Global Spacing Reduction */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }}
    
    /* Remove default top margin for titles */
    h1, h2, h3 {{ margin-top: 0 !important; }}
    
    /* Global Font */
    .stApp, h1, h2, h3, p, div, span {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }}
    
    /* Product Card */
    .product-card {{
        background-color: {current_theme['card_bg']};
        border: 1px solid rgba(0,0,0,.05);
        border-radius: 4px;
        box-shadow: 0 1px 1px 0 rgba(0,0,0,.05);
        transition: transform 0.1s;
        height: 320px;
        display: flex;
        flex-direction: column;
        cursor: pointer;
        overflow: hidden;
    }}
    .product-card:hover {{
        border: 1px solid {SHOP_ORANGE};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }}
    
    .product-img-container {{
        width: 100%;
        height: 180px;
        background-color: #fafafa;
        display: flex;
        align-items: center;
        justify-content: center;
        border-bottom: 1px solid #f1f1f1;
    }}
    
    .product-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
    
    .card-content {{
        padding: 8px;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    
    .product-title {{
        font-size: 13px;
        color: {current_theme['text_color']};
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2; 
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 4px;
        height: 34px;
    }}
    
    .product-price {{
        font-size: 15px;
        color: {current_theme['price_color']};
        font-weight: 600;
    }}
    
    .product-rating {{
        font-size: 11px;
        color: #757575;
        display: flex;
        align-items: center;
        margin-top: 4px;
    }}
    
    /* Login Box */
    .login-container {{
        background-color: white;
        padding: 40px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        max-width: 400px;
        margin: 0 auto;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'shop_sense_data.pkl')
USERS_FILE = os.path.join(BASE_DIR, 'users_db.json')

@st.cache_resource
def load_resources():
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

data_pack = load_resources()

if data_pack:
    df = data_pack['dataframe']
    tfidf_matrix = data_pack['tfidf_matrix']
    knn_model = data_pack['knn_model']
    user_item_matrix = data_pack['user_item_matrix']
    interactions_df = data_pack['interactions']
    
    # Ensure numerical types
    for col in ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
else:
    st.warning("⏳ Data generation in progress or file missing...")
    st.stop()

# --- USER AUTH & DB ---
import json

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try: return json.load(f)
        except: return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def register_user(username):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    # Assign a new loose ID for matrix ops
    new_id = np.random.randint(2000, 100000) 
    users[username] = {
        'id': new_id, 
        'history': [],
        'cart': []
    }
    save_users(users)
    return True, new_id

def authenticate_user(username):
    users = load_users()
    if username in users:
        # Return ID, History, Cart
        u_data = users[username]
        return True, u_data['id'], u_data.get('history', []), u_data.get('cart', [])
    return False, None, [], []

def update_user_db():
    """
    Persist current session state (History, Cart) to the JSON DB.
    """
    if st.session_state.user_id and st.session_state.user_id != "Guest":
        users = load_users()
        if st.session_state.user_id in users:
            users[st.session_state.user_id]['history'] = st.session_state.history
            users[st.session_state.user_id]['cart'] = st.session_state.cart
            save_users(users)

def record_interaction(user_id, product_idx, rating):
    """
    Real-time interaction recording + Persistence
    """
    global interactions_df
    new_row = {'user_id': user_id, 'product_index': product_idx, 'rating': rating}
    interactions_df = pd.concat([interactions_df, pd.DataFrame([new_row])], ignore_index=True)
    data_pack['interactions'] = interactions_df 
    
    # Also trigger DB Save for history/cart sync if needed
    update_user_db() 

# --- STATE ---
if 'page' not in st.session_state: st.session_state.page = 'login'
if 'cart' not in st.session_state: st.session_state.cart = [] # List of dicts
if 'user_id' not in st.session_state: st.session_state.user_id = None # Display Name
if 'sim_id' not in st.session_state: st.session_state.sim_id = None # Numeric ID for Matrix
if 'selected_product' not in st.session_state: st.session_state.selected_product = None
if 'selected_product' not in st.session_state: st.session_state.selected_product = None

# --- RECOMMENDATION LOGIC ---
def get_cbf_recs(idx, n=6):
    try:
        if idx is None: return pd.DataFrame()
        cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:n+1]
        indices = [i[0] for i in sim_scores]
        return df.iloc[indices]
    except: return pd.DataFrame()

def get_hybrid_recs(uid, history_items=None, n=12):
    try:
        recs_idx = []
        seen = set()
        
        if history_items:
            last_viewed = history_items[-1]
            try:
                # A. Content-Based (Visual/Description Similarity)
                cosine_sim = linear_kernel(tfidf_matrix[last_viewed:last_viewed+1], tfidf_matrix).flatten()
                sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:4] 
                for i_idx, score in sim_scores:
                    if i_idx not in seen:
                        recs_idx.append(i_idx)
                        seen.add(i_idx)
                        
                # B. Real-Time Collaborative (Item-User-Item)
                # "Find users who liked what I just viewed, and show me what else they liked."
                peers = interactions_df[
                    (interactions_df['product_index'] == last_viewed) & 
                    (interactions_df['rating'] >= 4.0)
                ]['user_id'].unique()
                
                # Limit to 5 peers for speed
                for peer in peers[:5]:
                    peer_likes = interactions_df[
                        (interactions_df['user_id'] == peer) & 
                        (interactions_df['rating'] >= 4.0)
                    ]['product_index'].values
                    
                    for p_idx in peer_likes:
                        if p_idx not in seen and p_idx != last_viewed:
                            recs_idx.append(p_idx)
                            seen.add(p_idx)
                            if len(recs_idx) >= 8: break # Cap peer recs
                    if len(recs_idx) >= 8: break
            except: pass

        # 2. Collaborative Filtering (Background Preferences)
        if uid is not None:
             try:
                user_vec = user_item_matrix.getrow(uid)
                dists, indices = knn_model.kneighbors(user_vec, n_neighbors=6)
                sim_users = indices[0][1:]
                for u in sim_users:
                    liked = interactions_df[(interactions_df['user_id'] == u) & (interactions_df['rating'] > 3.5)]
                    for p_idx in liked['product_index'].values:
                        if p_idx not in seen and p_idx not in recs_idx:
                            recs_idx.append(p_idx)
                            seen.add(p_idx)
                            if len(recs_idx) >= n: break
                    if len(recs_idx) >= n: break
             except: pass
             
        while len(recs_idx) < n:
            rand_idx = np.random.randint(0, len(df))
            if rand_idx not in seen and rand_idx not in recs_idx:
                recs_idx.append(rand_idx)
                
        return df.iloc[recs_idx[:n]]
    except: return pd.DataFrame()

# --- NAVIGATION ACTIONS ---
def go_home():
    st.session_state.page = 'home'
    st.session_state.selected_product = None

def go_product(idx):
    st.session_state.page = 'product'
    st.session_state.selected_product = idx
    if 'history' not in st.session_state: st.session_state.history = []
    st.session_state.history.append(idx)
    
    # Record VIEW interaction & Persist History
    if st.session_state.sim_id:
        record_interaction(st.session_state.sim_id, idx, 1.0)
        # record_interaction calls update_user_db() internally now, so redundant call removed

def go_cart():
    st.session_state.page = 'cart'

def render_sidebar():
    with st.sidebar:
        st.markdown(f"<h2 style='text-align:center; color:{SHOP_ORANGE}'>🛍️ ShopSense</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.session_state.user_id:
            st.markdown(f"<div style='text-align:center; margin-bottom:20px'><b>{st.session_state.user_id}</b></div>", unsafe_allow_html=True)
            
            if st.button("🏠 Home", use_container_width=True):
                go_home()
                st.rerun()
                
            cart_count = len(st.session_state.cart)
            if st.button(f"🛒 Cart ({cart_count})", use_container_width=True):
                go_cart()
                st.rerun()
                
            st.markdown("---")
            if st.button("Logout", use_container_width=True):
                update_user_db() # Persist current state before clearing
                st.session_state.history = []
                st.session_state.cart = []
                st.session_state.user_id = None
                st.session_state.sim_id = None
                st.session_state.selected_product = None
                st.session_state.page = 'login'
                st.rerun()
        else:
            st.info("Guest Mode")
            if st.button("Log In", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()

# --- VIEWS ---

def render_grid(products, title="", key_prefix="grid"):
    if title: st.subheader(title)
    if products.empty:
        st.info("No products found.")
        return

    cols = st.columns(5)
    for i, (idx, row) in enumerate(products.iterrows()):
        with cols[i % 5]:
            img_src = row['image']
            fallback = "https://via.placeholder.com/200"
            price = f"RM {float(row['discount_price']):.2f}"
            
            r_val = min(max(float(row['ratings']), 0.0), 5.0)
            
            st.markdown(f"""
            <div class="product-card">
                <div class="product-img-container">
                    <img src="{img_src}" class="product-img" onerror="this.onerror=null;this.src='{fallback}';">
                </div>
                <div class="card-content">
                    <div class="product-title">{row['name']}</div>
                    <div style="margin-top:auto">
                        <div class="product-price"><span style="font-size:10px">RM</span>{price[3:]}</div>
                        <div class="product-rating">
                            <span style="color:#ffce3d;">★</span> {r_val:.1f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View", key=f"btn_{key_prefix}_{idx}", use_container_width=True):
                go_product(idx)
                st.rerun()

def page_cart():
    render_sidebar()
    st.title("Shopping Cart 🛒")
    
    if not st.session_state.cart:
        st.info("Your cart is empty.")
        if st.button("Start Shopping"):
            go_home()
            st.rerun()
        return

    c1, c2 = st.columns([2, 1])
    
    total = 0.0
    
    with c1:
        st.subheader(f"Items ({len(st.session_state.cart)})")
        for i, item in enumerate(st.session_state.cart):
             total += item['price']
             st.markdown(f"""
             <div style="padding:15px; background:white; border-radius:8px; margin-bottom:10px; display:flex; align-items:center; box-shadow:0 1px 2px rgba(0,0,0,0.05)">
                <img src="{item['image']}" style="width:50px; height:50px; object-fit:contain; margin-right:15px; border-radius:4px;">
                <div style="flex:1;"><b>{item['name']}</b> <br> <span style="color:#ee4d2d">RM {item['price']:.2f}</span></div>
                <div>Qty: 1</div>
             </div>
             """, unsafe_allow_html=True)
            
    with c2:
        st.markdown(f"""
        <div style="background:white; padding:20px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
            <h4 style="margin-top:0">Order Summary</h4>
            <div style='font-size:24px; font-weight:bold; margin:20px 0; color:#ee4d2d'>RM {total:.2f}</div>
            <p style="font-size:12px; color:#666">Estimated Total</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        st.selectbox("Payment Method", ["Credit Card", "Online Banking", "COD"])
        
        if st.button("Place Order", type="primary", use_container_width=True):
            # Record Purchase Interaction (Strong Signal)
            if st.session_state.sim_id:
                for item in st.session_state.cart:
                    record_interaction(st.session_state.sim_id, item['idx'], 5.0)
            
            st.success("Order Placed Successfully!")
            st.balloons()
            st.session_state.cart = []
            update_user_db() # Persist empty cart
            time.sleep(2)
            go_home()
            st.rerun()

def page_product_detail():
    render_sidebar()
    idx = st.session_state.get('selected_product')
    if idx is None or idx not in df.index:
        go_home()
        st.rerun()
        return

    row = df.loc[idx]
    
    # Minimalist Header with Back Button
    c_back, c_title = st.columns([1, 10])
    with c_back:
        if st.button("⬅", help="Back to Home"):
            go_home()
            st.rerun()

    # Compact Layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown(f"""
        <div style="padding:10px; background:white; border-radius:8px; display:flex; justify-content:center;">
            <img src="{row['image']}" style="max-width:100%; max-height:350px; object-fit:contain;">
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h2 style='margin-bottom:10px'>{row['name']}</h2>", unsafe_allow_html=True)
        
        price_val = float(row['discount_price'])
        
        # Compact Rating & Price
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:15px; margin-bottom:20px'>
            <span style='font-size:24px; font-weight:bold; color:{SHOP_ORANGE}'>RM {price_val:.2f}</span>
            <span style='text-decoration:line-through; color:#999'>RM {float(row['actual_price']):.2f}</span>
            <span style='background:#fff0f0; color:{SHOP_ORANGE}; padding:2px 6px; font-size:12px; border-radius:4px'>On Sale</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px; font-size:14px; color:#555;'>
            <span>⭐ {float(row['ratings']):.1f}</span> • 
            <span>{int(row['no_of_ratings'])} ratings</span> • 
            <span>{np.random.randint(50, 2000)} sold</span>
        </div>
        <hr style='margin:15px 0'>
        """, unsafe_allow_html=True)
        
        # Description
        with st.expander("Product Description", expanded=True):
            st.write(f"Category: {row['main_category']} > {row['category']}")
            st.caption("Authentic quality product. Fast delivery. 15-day return policy.")
            
        c_act1, c_act2 = st.columns([1, 2])
        with c_act2:
            if st.button("Add to Cart", type="primary", use_container_width=True):
                # Add actual item data to cart
                item_data = {
                    'idx': idx,
                    'name': row['name'],
                    'price': price_val,
                    'image': row['image']
                }
                st.session_state.cart.append(item_data)
                
                # Record Cart Interaction & Persist Cart
                if st.session_state.sim_id:
                    record_interaction(st.session_state.sim_id, idx, 3.5)
                update_user_db() # Persist cart after adding item
                
                st.toast("Added to Cart!")

    # Compact Recommendations
    st.write("")
    st.write("")
    st.markdown("##### You might also like")
    recs = get_cbf_recs(idx, n=5)
    if not recs.empty:
        render_grid(recs, "", key_prefix="sim")

def page_login():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Logo Substitute
        st.markdown(f"<div style='font-size:60px; text-align:center;'>🛍️</div>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color:{SHOP_ORANGE}; margin-bottom:10px; text-align:center;'>ShopSense</h1>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.write("")
            username = st.text_input("Username", key="login_user")
            
            if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
                if username:
                    # Unpack 4 values: success, id, history, cart
                    success, uid, hist, cart = authenticate_user(username)
                    if success:
                        st.success(f"Welcome back, {username}!")
                        st.session_state.user_id = username
                        st.session_state.sim_id = uid
                        st.session_state.page = 'home'
                        st.session_state.cart = cart
                        st.session_state.history = hist
                        st.rerun()
                    else:
                        st.error("User not found due to invalid credentials.")
                else:
                    st.error("Please enter username.")

        with tab2:
            st.write("")
            new_user = st.text_input("Choose Username", key="signup_user")
            if st.button("Sign Up", use_container_width=True, key="btn_signup"):
                if new_user:
                    success, uid = register_user(new_user)
                    if success:
                        st.success("Account created! Logging in...")
                        st.session_state.user_id = new_user
                        st.session_state.sim_id = uid
                        st.session_state.page = 'home'
                        st.session_state.cart = []
                        st.session_state.history = []
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(uid) # Error message
                else:
                    st.error("Please enter a username.")
        
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.user_id = "Guest"
            st.session_state.sim_id = None
            st.session_state.page = 'home'
            st.session_state.cart = []
            st.session_state.history = []
            st.rerun()

def page_home():
    render_sidebar()
    
    # Search Bar
    search = st.text_input("", placeholder="Search for products...", label_visibility="collapsed")
    
    # Sidebar Filters
    with st.sidebar:
        st.markdown("### Filters")
        if 'category' in df.columns:
            cats = ["All"] + list(sorted(df['category'].unique()))
            selected_cat = st.selectbox("Category", cats, index=0)
        else:
            selected_cat = "All"
            
    filtered = df.copy()
    if search:
        filtered = filtered[filtered['name'].str.contains(search, case=False)]
    if selected_cat != "All":
        filtered = filtered[filtered['category'] == selected_cat]
        
    # Recommendations
    history = st.session_state.get('history', [])
    
    # Initialize caching for stable buttons
    if 'home_recs' not in st.session_state: st.session_state.home_recs = pd.DataFrame()
    if 'last_hist_len' not in st.session_state: st.session_state.last_hist_len = 0
    
    # Update recs only if history changed or we have none
    if len(history) != st.session_state.last_hist_len or st.session_state.home_recs.empty:
         if not search and selected_cat == "All":
             new_recs = get_hybrid_recs(st.session_state.sim_id, history_items=history, n=10)
             st.session_state.home_recs = new_recs
             st.session_state.last_hist_len = len(history)

    if not search and selected_cat == "All":
        recs = st.session_state.home_recs
        if not recs.empty:
             render_grid(recs, f"Recommended For You {'(Updated)' if history else ''}", key_prefix="rec")
             st.markdown("---")
             
    title = f"{selected_cat}" if selected_cat != "All" else "Daily Discover"
    render_grid(filtered.head(40), title, key_prefix="main")

# --- ROUTER ---
if st.session_state.page == 'login': page_login()
elif st.session_state.page == 'home': page_home()
elif st.session_state.page == 'product': page_product_detail()
elif st.session_state.page == 'cart': page_cart()
