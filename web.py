import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):
    np.bool = bool

def setup_chinese_font():
    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")

        # 兜底：使用英文字体（中文将显示为方框）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="基于XGBoost算法预测早发心肌梗死后心力衰竭风险的网页计算器",
    page_icon="🏥",
    layout="wide"
)


if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


feature_names_display = [
    'Outcome_CHD_DM',      # 糖尿病（0/1）
    'Outcome_feiyan',      # 肺部感染（0/1）
    'Tachyarrhythmia',     # 快速性心律失常（0/1）
    'TCM',                 # 中药干预（0/1）
    'Qizhixueyu',          # 气滞血瘀（0/1）
    'Yangxu',              # 阳虚（0/1）
    'Xueyushuiting',       # 血瘀水停（0/1）
    'Age',                 # 年龄（岁）
    'Pulse_rate',          # 心率（次/分）
    'Hb',                  # 血红蛋白（g/L）
    'SCr',                 # 血清肌酐（μmol/L）
    'BUN'                  # 血尿素氮（mmol/L）
]

# 中文显示名称
feature_names_cn = [
    '糖尿病', '肺部感染', '快速性心律失常', '中药干预',
    '气滞血瘀', '阳虚', '血瘀水停',
    '年龄', '心率', '血红蛋白', '血清肌酐', '血尿素氮'
]

feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明（鼠标悬停提示用）
variable_descriptions = {
    'Outcome_CHD_DM':  '有无糖尿病（0=无，1=有）',
    'Outcome_feiyan':  '有无肺部感染（0=无，1=有）',
    'Tachyarrhythmia': '有无快速性心律失常（0=无，1=有）',
    'TCM':             '是否接受中药干预（0=无，1=有）',
    'Qizhixueyu':      '有无气滞血瘀（0=无，1=有）',
    'Yangxu':          '有无阳虚（0=无，1=有）',
    'Xueyushuiting':   '有无血瘀水停（0=无，1=有）',
    'Age':             '年龄（岁）',
    'Pulse_rate':      '心率（次/分）',
    'Hb':              '血红蛋白（g/L）',
    'SCr':             '血清肌酐（μmol/L）',
    'BUN':             '血尿素氮（mmol/L）'
}

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # ---------- 侧边栏 ----------
    st.sidebar.title("早发心肌梗死后心力衰竭风险预测计算器")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)
    st.sidebar.markdown("""
    # 系统说明
    本系统基于 XGBoost 算法，通过临床指标预测 **早发心肌梗死后心力衰竭** 的发生风险。

    ## 预测输出
    - 心力衰竭发生概率
    - 未发生心力衰竭概率
    - 风险分层（低 / 中 / 高）

    ## 使用方法
    1. 填写下方全部指标
    2. 点击“开始预测”
    3. 查看结果与 SHAP 解释
    """)

    with st.sidebar.expander("变量说明"):
        for f in feature_names_display:
            st.markdown(f"**{feature_dict[f]}**: {variable_descriptions[f]}")

    # ---------- 主页面 ----------
    st.title("早发心肌梗死后心力衰竭风险预测计算器")
    st.markdown("### 请录入全部特征后点击预测")
    st.caption("单位：血红蛋白-g/L，血清肌酐-μmol/L，血尿素氮-mmol/L，心率-次/分")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return

    # ---------- 输入区域 ----------
    st.header("患者指标录入")
    col1, col2, col3 = st.columns(3)

    with col1:
        diabetes = st.selectbox("糖尿病", [0, 1], format_func=lambda x: "有" if x else "无")
        feiyan = st.selectbox("肺部感染", [0, 1], format_func=lambda x: "有" if x else "无")
        tachy = st.selectbox("快速性心律失常", [0, 1], format_func=lambda x: "有" if x else "无")
        tcm = st.selectbox("中药干预", [0, 1], format_func=lambda x: "有" if x else "无")

    with col2:
        qizhi = st.selectbox("气滞血瘀", [0, 1], format_func=lambda x: "有" if x else "无")
        yangxu = st.selectbox("阳虚", [0, 1], format_func=lambda x: "有" if x else "无")
        xueyu = st.selectbox("血瘀水停", [0, 1], format_func=lambda x: "有" if x else "无")
        age = st.number_input("年龄（岁）", value=55, step=1, min_value=18, max_value=100)

    with col3:
        pulse = st.number_input("心率（次/分）", value=80, step=1, min_value=40, max_value=200)
        hb = st.number_input("血红蛋白（g/L）", value=130, step=1)
        scr = st.number_input("血清肌酐（μmol/L）", value=80.0, step=0.1)
        bun = st.number_input("血尿素氮（mmol/L）", value=5.0, step=0.1)

    # ---------- 预测 ----------
    if st.button("开始预测", type="primary"):
        user_inputs = {
            'Outcome_CHD_DM': diabetes,
            'Outcome_feiyan': feiyan,
            'Tachyarrhythmia': tachy,
            'TCM': tcm,
            'Qizhixueyu': qizhi,
            'Yangxu': yangxu,
            'Xueyushuiting': xueyu,
            'Age': age,
            'Pulse_rate': pulse,
            'Hb': hb,
            'SCr': scr,
            'BUN': bun
        }

        if model_feature_names:
            lowered_features = [c.lower() for c in model_feature_names]
            missing = [c for c in lowered_features if c not in {k.lower(): v for k, v in user_inputs.items()}]
            if missing:
                st.error(f"缺失特征：{missing}")
                return
            input_df = pd.DataFrame([[user_inputs[c] for c in model_feature_names]],
                                    columns=model_feature_names)
        else:
            input_df = pd.DataFrame([user_inputs])[feature_names_display]

        if input_df.isnull().any().any():
            st.error("存在缺失值，请检查")
            return

        try:
            proba = model.predict_proba(input_df)[0]
            no_hf_prob = float(proba[0])
            hf_prob = float(proba[1])
        except Exception as e:
            st.error(f"预测失败: {e}")
            return

        # ---------- 结果展示 ----------
        st.header("心力衰竭风险预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("未发生概率")
            st.progress(no_hf_prob)
            st.write(f"{no_hf_prob:.2%}")
        with col2:
            st.subheader("发生概率")
            st.progress(hf_prob)
            st.write(f"{hf_prob:.2%}")

        risk_level = "低风险" if hf_prob < 0.3 else ("中等风险" if hf_prob < 0.7 else "高风险")
        risk_color = "green" if hf_prob < 0.3 else ("orange" if hf_prob < 0.7 else "red")
        st.markdown(f"### 风险评估：<span style='color:{risk_color}'>{risk_level}</span>",
                    unsafe_allow_html=True)

        # ---------- SHAP 解释 ----------
        st.write("---")
        st.subheader("模型解释（SHAP）")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_val = np.array(shap_values[1][0])
                ev = explainer.expected_value[1]
            else:
                shap_val = np.array(shap_values[0])
                ev = explainer.expected_value

            # 瀑布图
            fig = plt.figure(figsize=(12, 6))
            shap.waterfall_plot(
                shap.Explanation(values=shap_val,
                                 base_values=ev,
                                 data=input_df.iloc[0].values,
                                 feature_names=[feature_dict.get(f, f) for f in input_df.columns]),
                max_display=len(input_df.columns), show=False)
            st.pyplot(fig)
            plt.close(fig)

            # 力图
            import streamlit.components.v1 as components
            force_plot = shap.force_plot(ev, shap_val, input_df,
                                         feature_names=[feature_dict.get(f, f) for f in input_df.columns])
            components.html(force_plot.html(), height=400, scrolling=False)

        except Exception as e:
            st.error(f"SHAP 解释生成失败: {e}")

    st.write("---")
    st.caption(" ")

if __name__ == "__main__":
    main()

