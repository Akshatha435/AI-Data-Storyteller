# ===============================
# TAB 2: VISUAL ANALYTICS
# ===============================
with tab_visual:

    st.markdown("## Multivariate analysis")

    m1, m2 = st.columns(2)

    with m1:
        multi_chart = st.selectbox(
            "Chart type",
            [
                "Correlation heatmap",
                "Stacked bar chart",
                "Boxplot by category",
                "Scatter with hue",
                "Line chart",
                "Pairplot (numeric only)"
            ],
            key="multi_chart_type"
        )

    with m2:
        multi_cols = st.multiselect(
            "Select columns (2 or more)",
            df.columns,
            key="multi_chart_cols"
        )

    if len(multi_cols) >= 2:

        data = df[multi_cols].copy()
        fig = None  # IMPORTANT: reset per run

        # ---------- CORRELATION ----------
        if multi_chart == "Correlation heatmap":
            numeric_df = data.select_dtypes(include="number")

            if numeric_df.shape[1] < 2:
                st.warning("Select at least two numeric columns for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    numeric_df.corr(),
                    cmap="Blues",
                    center=0,
                    linewidths=0.5,
                    ax=ax
                )

        # ---------- STACKED BAR ----------
        elif multi_chart == "Stacked bar chart":
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.crosstab(
                data.iloc[:, 0],
                data.iloc[:, 1]
            ).plot(kind="bar", stacked=True, ax=ax)

        # ---------- BOXPLOT ----------
        elif multi_chart == "Boxplot by category":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df,
                x=multi_cols[0],
                y=multi_cols[1],
                ax=ax
            )

        # ---------- SCATTER ----------
        elif multi_chart == "Scatter with hue":
            fig, ax = plt.subplots(figsize=(6, 4))

            if len(multi_cols) >= 3:
                sns.scatterplot(
                    data=df,
                    x=multi_cols[0],
                    y=multi_cols[1],
                    hue=multi_cols[2],
                    ax=ax
                )
            else:
                sns.scatterplot(
                    data=df,
                    x=multi_cols[0],
                    y=multi_cols[1],
                    ax=ax
                )

        # ---------- LINE ----------
        elif multi_chart == "Line chart":
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df[multi_cols[0]], df[multi_cols[1]])

        # ---------- PAIRPLOT ----------
        elif multi_chart == "Pairplot (numeric only)":
            numeric_df = data.select_dtypes(include="number")

            if numeric_df.shape[1] < 2:
                st.warning("Select at least two numeric columns for pairplot.")
            else:
                pair_fig = sns.pairplot(numeric_df)
                st.pyplot(pair_fig.fig)
                fig = None  # pairplot handled separately

        if fig is not None:
            st.pyplot(fig)

            if st.button("Save multivariate chart to report", key="save_multi_chart"):
                img_path = save_figure(
                    fig,
                    f"multi_{multi_chart.replace(' ', '_')}"
                )
                st.session_state["report_visuals"].append(
                    {
                        "image": img_path,
                        "title": f"{multi_chart} – {', '.join(multi_cols)}",
                        "insight": ""
                    }
                )
                st.success("Multivariate chart saved.")
