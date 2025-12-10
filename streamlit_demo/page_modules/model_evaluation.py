"""
Model Training Evaluation page.
"""
import streamlit as st
import os
import sys
import tempfile
import shutil
import traceback
import pandas as pd
from datetime import datetime
_this_dir = os.path.dirname(__file__)
_parent_dir = os.path.abspath(os.path.join(_this_dir, '..'))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from utils import (
    list_models,
    get_model_by_id,
    get_preprocess_for_model,
    get_postprocess_for_model,
    upsert_model,
    solve_captcha,
    apply_preprocess,
    load_validation_dataset,
    evaluate_model,
    PLOTLY_AVAILABLE,
    go,
)

def render():
    """Render the Model Training Evaluation page."""
    st.header("Model Training Evaluation")
    st.info("Select a model from MongoDB and evaluate it using ground truth annotations.")
    
    # Model selection
    try:
        models = list_models(limit=100)
        if not models:
            st.warning("No models found in MongoDB. Please create a model first in the 'Create and Upload Model' section.")
        else:
            # Sort models by model_id in ascending order to ensure consistent dropdown ordering
            sorted_models = sorted(models, key=lambda m: m.get('model_id', ''))
            model_options = {f"{m.get('model_name', 'Unknown')} ({m.get('model_id', 'N/A')})": m.get('model_id') for m in sorted_models}
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            key="model_eval_select"
        )
        selected_model_id = model_options[selected_model_name]
        selected_model = get_model_by_id(selected_model_id)

        if selected_model:
            st.markdown("### Model Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model ID", selected_model.get('model_id', 'N/A'))
            with col2:
                st.metric("Model Name", selected_model.get('model_name', 'N/A'))
            with col3:
                st.metric("Active", "Yes" if selected_model.get('is_active', False) else "No")

            # Display existing metrics if available
            existing_results = selected_model.get('results', {})
            if existing_results:
                st.markdown("#### Existing Metrics (from database)")
                res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                with res_col1:
                    st.metric("Precision", f"{existing_results.get('precision', 0):.4f}" if existing_results.get('precision') else "N/A")
                with res_col2:
                    st.metric("Recall", f"{existing_results.get('recall', 0):.4f}" if existing_results.get('recall') else "N/A")
                with res_col3:
                    st.metric("F1 Score", f"{existing_results.get('f1_score', 0):.4f}" if existing_results.get('f1_score') else "N/A")
                with res_col4:
                    st.metric("mAP@0.5", f"{existing_results.get('mAP50', 0):.4f}" if existing_results.get('mAP50') else "N/A")
                with res_col5:
                    st.metric("mAP@0.5:0.95", f"{existing_results.get('AP5095', 0):.4f}" if existing_results.get('AP5095') else "N/A")

            st.markdown("---")

            st.info("Upload the data.yaml file from your Roboflow dataset. The validation split will be used for evaluation.")

            data_yaml_file = st.file_uploader(
                "Upload data.yaml file",
                type=["yaml", "yml"],
                key="data_yaml_upload_eval"
            )

            if data_yaml_file:
                if st.button("Run Evaluation", key="run_eval_button"):
                    with st.spinner("Loading validation dataset and running evaluation..."):
                        import tempfile
                        import shutil

                        # Create temporary directory for dataset
                        temp_dir = tempfile.mkdtemp()

                        try:
                                # Save data.yaml file
                                data_yaml_path = os.path.join(temp_dir, "data.yaml")
                                with open(data_yaml_path, 'wb') as f:
                                    f.write(data_yaml_file.read())

                                # Load validation dataset
                                st.info("Loading validation dataset from data.yaml...")
                                st.caption("Note: The evaluation process is READ-ONLY. Your original dataset files will NOT be modified.")
                                image_files, all_annotations, class_names = load_validation_dataset(data_yaml_path)

                                if not image_files:
                                    st.error("No validation images found. Please check your data.yaml file and dataset structure.")
                                else:
                                    st.success(f"Loaded {len(image_files)} validation images with annotations")

                                    # Flatten annotations for evaluation
                                    ground_truth = []
                                    for annotations in all_annotations:
                                        ground_truth.extend(annotations)

                                    st.info(f"Total ground truth annotations: {len(ground_truth)}")

                                    # Show ground truth class names from data.yaml
                                    if class_names:
                                        st.caption(f"Ground truth classes from data.yaml: {sorted([class_names[i] for i in sorted(class_names.keys())])}")

                                    # Prepare model config for direct inference (bypassing API validation)
                                    model_config = {
                                        'model_id': selected_model_id,
                                        'model_name': selected_model.get('model_name', ''),
                                    }

                                    # Get preprocessing and postprocessing profiles for the selected model
                                    preprocess_profile = get_preprocess_for_model(selected_model)
                                    postprocess_profile_retrieved = get_postprocess_for_model(selected_model)

                                    # Prepare postprocess profile for solve_captcha (full structure)
                                    postprocess_profile = None
                                    if postprocess_profile_retrieved:
                                        postprocess_profile = {
                                            'postprocess_id': postprocess_profile_retrieved.get('postprocess_id'),
                                            'name': postprocess_profile_retrieved.get('name'),
                                            'steps': postprocess_profile_retrieved.get('steps', [])
                                        }

                                    # Run inference directly on validation images (no API, no validation)
                                    all_predictions = []

                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    for idx, img_path in enumerate(image_files):
                                        status_text.text(f"Processing image {idx + 1}/{len(image_files)}: {os.path.basename(img_path)}")
                                        progress_bar.progress((idx + 1) / len(image_files))

                                        try:
                                            # Use native YOLOv8 evaluation mode to match training evaluation exactly
                                            # This skips preprocessing and uses file paths directly, matching YOLOv8's native behavior
                                            inference_result = solve_captcha(
                                                img_path,  # Pass file path directly for native evaluation
                                                question="",  # Empty question - bypasses validation
                                                config=model_config,
                                                postprocess_profile=None,  # Skip postprocessing for native evaluation
                                                use_native_eval=True,  # Enable native evaluation mode
                                                imgsz=None  # Use model's default image size (matches training)
                                            )

                                            # Handle inference result
                                            if isinstance(inference_result, dict):
                                                if 'error' in inference_result:
                                                    st.warning(f"Inference error for {os.path.basename(img_path)}: {inference_result['error']}")
                                                    continue
                                                elif 'message' in inference_result:
                                                    # No detections
                                                    detections = []
                                                else:
                                                    detections = inference_result
                                            elif isinstance(inference_result, list):
                                                detections = inference_result
                                            else:
                                                detections = []

                                            # Add image identifier to each detection
                                            for det in detections:
                                                if isinstance(det, dict):
                                                    det['image_id'] = os.path.basename(img_path)

                                            all_predictions.extend(detections)
                                        except Exception as e:
                                            st.warning(f"Failed to run inference on {os.path.basename(img_path)}: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())

                                    progress_bar.empty()
                                    status_text.empty()

                                    if all_predictions:
                                        st.success(f"Generated {len(all_predictions)} predictions from {len(image_files)} validation images")

                                        # Debug: Show class names from predictions and ground truth
                                        pred_classes = set([p.get('class', '') for p in all_predictions if isinstance(p, dict)])
                                        gt_classes = set([g.get('class', '') for g in ground_truth if isinstance(g, dict)])

                                        with st.expander("Debug: Class Name Comparison"):
                                            st.write("**Prediction Classes:**", sorted(pred_classes))
                                            st.write("**Ground Truth Classes:**", sorted(gt_classes))
                                            st.write("**Matching Classes:**", sorted(pred_classes & gt_classes))
                                            st.write("**Only in Predictions:**", sorted(pred_classes - gt_classes))
                                            st.write("**Only in Ground Truth:**", sorted(gt_classes - pred_classes))

                                        # Create class mapping if needed (map model class names to ground truth class names)
                                        # This handles cases where model uses different class names than ground truth
                                        class_mapping = {}
                                        if pred_classes != gt_classes:
                                            st.warning("Class name mismatch detected! Attempting automatic mapping...")
                                            # Try to match by similarity (case-insensitive, underscore/space normalization)
                                            for pred_class in pred_classes:
                                                pred_normalized = pred_class.lower().replace('_', ' ').replace('-', ' ')
                                                for gt_class in gt_classes:
                                                    gt_normalized = gt_class.lower().replace('_', ' ').replace('-', ' ')
                                                    if pred_normalized == gt_normalized:
                                                        class_mapping[pred_class] = gt_class
                                                        break

                                            if class_mapping:
                                                st.info(f"Created class mapping: {class_mapping}")
                                            else:
                                                st.error("Could not automatically map class names. Please ensure model and dataset use the same class names.")

                                        # Run evaluation with class mapping
                                        eval_results = evaluate_model(
                                            all_predictions,
                                            ground_truth,
                                            iou_threshold=0.5,
                                            class_mapping=class_mapping if class_mapping else None
                                        )
                                        
                                        # Store predictions and ground truth for curve generation
                                        eval_results['all_predictions'] = all_predictions
                                        eval_results['ground_truth'] = ground_truth
                                        eval_results['class_mapping'] = class_mapping if class_mapping else None

                                        # Display overall metrics
                                        st.markdown("### Overall Evaluation Metrics")
                                        overall = eval_results['overall_metrics']

                                        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                                        with metric_col1:
                                            st.metric("mAP@0.5", f"{overall['map_50']:.4f}")
                                        with metric_col2:
                                            st.metric("mAP@0.5:0.95", f"{overall['map_50_95']:.4f}")
                                        with metric_col3:
                                            st.metric("Total TP", overall['total_tp'])
                                        with metric_col4:
                                            st.metric("Total FP", overall['total_fp'])
                                        with metric_col5:
                                            st.metric("Total FN", overall['total_fn'])

                                        # Calculate macro-averaged precision, recall, and F1 from per-class metrics
                                        per_class = eval_results['per_class_metrics']
                                        macro_precision = 0.0
                                        macro_recall = 0.0
                                        macro_f1 = 0.0
                                        if per_class:
                                            precisions = [m['precision'] for m in per_class.values()]
                                            recalls = [m['recall'] for m in per_class.values()]
                                            f1_scores = [m['f1_score'] for m in per_class.values()]
                                            macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
                                            macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
                                            macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

                                        # Prepare results for database update
                                        # Format matches expected structure: {precision, recall, f1_score, mAP50, AP5095}
                                        db_results = {
                                            'precision': macro_precision,
                                            'recall': macro_recall,
                                            'f1_score': macro_f1,
                                            'mAP50': overall['map_50'],
                                            'AP5095': overall['map_50_95'],
                                            'total_tp': overall['total_tp'],
                                            'total_fp': overall['total_fp'],
                                            'total_fn': overall['total_fn'],
                                            'evaluation_timestamp': datetime.now().isoformat(),
                                            'per_class_metrics': per_class  # Store detailed per-class metrics
                                        }

                                        # Update model in database with evaluation results
                                        try:
                                            updated_model = upsert_model(
                                                model_id=selected_model_id,
                                                model_name=selected_model.get('model_name', ''),
                                                weights_file_stream=None,  # Don't update weights
                                                results=db_results,
                                                is_active=selected_model.get('is_active', False)  # Preserve is_active status
                                            )
                                            if updated_model:
                                                st.success("Evaluation results saved to database!")
                                            else:
                                                st.warning("Could not save evaluation results to database. Please check database connection.")
                                        except Exception as e:
                                            st.error(f"Error saving evaluation results: {e}")
                                            import traceback
                                            st.code(traceback.format_exc())

                                        # Per-class metrics table
                                        st.markdown("### Per-Class Metrics")
                                        per_class = eval_results['per_class_metrics']

                                        if per_class:
                                            class_data = []
                                            for class_name, metrics in per_class.items():
                                                class_data.append({
                                                    'Class': class_name,
                                                    'Precision': f"{metrics['precision']:.4f}",
                                                    'Recall': f"{metrics['recall']:.4f}",
                                                    'F1 Score': f"{metrics['f1_score']:.4f}",
                                                    'AP@0.5': f"{metrics['ap_50']:.4f}",
                                                    'TP': metrics['tp'],
                                                    'FP': metrics['fp'],
                                                    'FN': metrics['fn']
                                                })

                                            df_metrics = pd.DataFrame(class_data)
                                            st.dataframe(df_metrics, width='stretch')

                                            # Per-class metrics charts
                                            st.markdown("#### Per-Class Metrics Visualization")

                                            if PLOTLY_AVAILABLE:
                                                classes = list(per_class.keys())
                                                precisions = [per_class[c]['precision'] for c in classes]
                                                recalls = [per_class[c]['recall'] for c in classes]
                                                f1_scores = [per_class[c]['f1_score'] for c in classes]

                                                # Precision, Recall, F1 chart
                                                fig_prf = go.Figure()
                                                fig_prf.add_trace(go.Bar(
                                                    name='Precision',
                                                    x=classes,
                                                    y=precisions,
                                                    marker_color='lightblue'
                                                ))
                                                fig_prf.add_trace(go.Bar(
                                                    name='Recall',
                                                    x=classes,
                                                    y=recalls,
                                                    marker_color='lightgreen'
                                                ))
                                                fig_prf.add_trace(go.Bar(
                                                    name='F1 Score',
                                                    x=classes,
                                                    y=f1_scores,
                                                    marker_color='lightcoral'
                                                ))
                                                fig_prf.update_layout(
                                                    title='Per-Class Precision, Recall, and F1 Score',
                                                    xaxis_title='Class',
                                                    yaxis_title='Score',
                                                    barmode='group',
                                                    height=400,
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(fig_prf, width='stretch')

                                                # AP@0.5 chart
                                                fig_ap = go.Figure()
                                                aps = [per_class[c]['ap_50'] for c in classes]
                                                fig_ap.add_trace(go.Bar(
                                                    x=classes,
                                                    y=aps,
                                                    marker_color='steelblue'
                                                ))
                                                fig_ap.update_layout(
                                                    title='Per-Class AP@0.5',
                                                    xaxis_title='Class',
                                                    yaxis_title='AP@0.5',
                                                    height=400,
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(fig_ap, width='stretch')

                                                # TP, FP, FN chart
                                                fig_counts = go.Figure()
                                                tps = [per_class[c]['tp'] for c in classes]
                                                fps = [per_class[c]['fp'] for c in classes]
                                                fns = [per_class[c]['fn'] for c in classes]
                                                fig_counts.add_trace(go.Bar(
                                                    name='True Positives',
                                                    x=classes,
                                                    y=tps,
                                                    marker_color='green'
                                                ))
                                                fig_counts.add_trace(go.Bar(
                                                    name='False Positives',
                                                    x=classes,
                                                    y=fps,
                                                    marker_color='red'
                                                ))
                                                fig_counts.add_trace(go.Bar(
                                                    name='False Negatives',
                                                    x=classes,
                                                    y=fns,
                                                    marker_color='orange'
                                                ))
                                                fig_counts.update_layout(
                                                    title='Per-Class Detection Counts (TP, FP, FN)',
                                                    xaxis_title='Class',
                                                    yaxis_title='Count',
                                                    barmode='group',
                                                    height=400,
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(fig_counts, width='stretch')

                                                # Overall metrics summary chart
                                                st.markdown("#### Overall Metrics Summary")
                                                summary_fig = go.Figure()
                                                summary_fig.add_trace(go.Bar(
                                                    x=['mAP@0.5', 'mAP@0.5:0.95'],
                                                    y=[overall['map_50'], overall['map_50_95']],
                                                    marker_color='purple',
                                                    text=[f"{overall['map_50']:.4f}", f"{overall['map_50_95']:.4f}"],
                                                    textposition='auto'
                                                ))
                                                summary_fig.update_layout(
                                                    title='Overall mAP Metrics',
                                                    xaxis_title='Metric',
                                                    yaxis_title='Score',
                                                    height=400,
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(summary_fig, width='stretch')
                                                
                                                # Advanced Evaluation Curves
                                                st.markdown("---")
                                                st.markdown("### Advanced Evaluation Curves")
                                                
                                                # Generate curves
                                                from app.evaluator import match_predictions_to_ground_truth, calculate_precision_recall_f1
                                                import numpy as np
                                                
                                                # Get all unique classes
                                                all_classes = set()
                                                for pred in all_predictions:
                                                    class_name = pred.get('class', '')
                                                    if class_mapping and class_name in class_mapping:
                                                        class_name = class_mapping[class_name]
                                                    if class_name:
                                                        all_classes.add(class_name)
                                                for gt in ground_truth:
                                                    class_name = gt.get('class', '')
                                                    if class_name:
                                                        all_classes.add(class_name)
                                                
                                                # Confidence thresholds for curves
                                                conf_thresholds = np.arange(0.0, 1.01, 0.01)
                                                
                                                # 1. Precision-Recall Curve
                                                st.markdown("#### Precision-Recall Curve")
                                                pr_fig = go.Figure()
                                                
                                                # Calculate PR curve for each class
                                                class_aps = {}
                                                for class_name in sorted(all_classes):
                                                    precisions = []
                                                    recalls = []
                                                    
                                                    for conf_thresh in conf_thresholds:
                                                        # Filter predictions by confidence threshold
                                                        filtered_preds = [
                                                        p for p in all_predictions
                                                        if p.get('confidence', 0) >= conf_thresh
                                                        and (class_mapping.get(p.get('class', ''), p.get('class', '')) if class_mapping else p.get('class', '')) == class_name
                                                    ]
                                                        
                                                        # Match predictions to ground truth
                                                        tp, fp, fn = match_predictions_to_ground_truth(
                                                            filtered_preds,
                                                            [g for g in ground_truth if g.get('class') == class_name],
                                                            iou_threshold=0.5,
                                                            class_mapping=class_mapping
                                                        )
                                                        
                                                        # Calculate precision and recall
                                                        tp_count = len([x for x in tp if x.get('class') == class_name])
                                                        fp_count = len([x for x in fp if x.get('class') == class_name])
                                                        fn_count = len([x for x in fn if x.get('class') == class_name])
                                                        
                                                        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                                                        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
                                                        
                                                        precisions.append(precision)
                                                        recalls.append(recall)
                                                    
                                                    # Calculate AP (Area Under PR Curve)
                                                    ap = np.trapz(precisions, recalls) if len(recalls) > 1 and len(precisions) > 1 else 0.0
                                                    class_aps[class_name] = ap
                                                    
                                                    # Add trace for this class
                                                    pr_fig.add_trace(go.Scatter(
                                                        x=recalls,
                                                        y=precisions,
                                                        mode='lines',
                                                        name=f"{class_name} (AP={ap:.3f})",
                                                        line=dict(width=2)
                                                    ))
                                                
                                                # Add overall mAP curve
                                                overall_precisions = []
                                                overall_recalls = []
                                                for conf_thresh in conf_thresholds:
                                                    filtered_preds = [p for p in all_predictions if p.get('confidence', 0) >= conf_thresh]
                                                    tp, fp, fn = match_predictions_to_ground_truth(
                                                        filtered_preds, ground_truth, iou_threshold=0.5, class_mapping=class_mapping
                                                    )
                                                    total_tp = len(tp)
                                                    total_fp = len(fp)
                                                    total_fn = len(fn)
                                                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                                                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                                                    overall_precisions.append(precision)
                                                    overall_recalls.append(recall)
                                                
                                                map_50 = overall['map_50']
                                                pr_fig.add_trace(go.Scatter(
                                                    x=overall_recalls,
                                                    y=overall_precisions,
                                                    mode='lines',
                                                    name=f"all classes (mAP@0.5={map_50:.3f})",
                                                    line=dict(width=3, color='black')
                                                ))
                                                
                                                pr_fig.update_layout(
                                                    title='Precision-Recall Curve',
                                                    xaxis_title='Recall',
                                                    yaxis_title='Precision',
                                                    xaxis=dict(range=[0, 1]),
                                                    yaxis=dict(range=[0, 1]),
                                                    height=500,
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(pr_fig, width='stretch')
                                                
                                                # 2. F1-Confidence Curve
                                                st.markdown("#### F1-Confidence Curve")
                                                f1_fig = go.Figure()
                                                
                                                for class_name in sorted(all_classes):
                                                    f1_scores = []
                                                    for conf_thresh in conf_thresholds:
                                                        filtered_preds = [
                                                            p for p in all_predictions
                                                            if p.get('confidence', 0) >= conf_thresh
                                                            and (class_mapping.get(p.get('class', ''), p.get('class', '')) if class_mapping else p.get('class', '')) == class_name
                                                        ]
                                                        tp, fp, fn = match_predictions_to_ground_truth(
                                                            filtered_preds,
                                                            [g for g in ground_truth if g.get('class') == class_name],
                                                            iou_threshold=0.5,
                                                            class_mapping=class_mapping
                                                        )
                                                        tp_count = len([x for x in tp if x.get('class') == class_name])
                                                        fp_count = len([x for x in fp if x.get('class') == class_name])
                                                        fn_count = len([x for x in fn if x.get('class') == class_name])
                                                        
                                                        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                                                        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
                                                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                                                        f1_scores.append(f1)
                                                    
                                                    f1_fig.add_trace(go.Scatter(
                                                        x=conf_thresholds,
                                                        y=f1_scores,
                                                        mode='lines',
                                                        name=class_name,
                                                        line=dict(width=2)
                                                    ))
                                                
                                                # Overall F1 curve
                                                overall_f1_scores = []
                                                for conf_thresh in conf_thresholds:
                                                    filtered_preds = [p for p in all_predictions if p.get('confidence', 0) >= conf_thresh]
                                                    tp, fp, fn = match_predictions_to_ground_truth(
                                                        filtered_preds, ground_truth, iou_threshold=0.5, class_mapping=class_mapping
                                                    )
                                                    total_tp = len(tp)
                                                    total_fp = len(fp)
                                                    total_fn = len(fn)
                                                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                                                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                                                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                                                    overall_f1_scores.append(f1)
                                                
                                                # Find best F1 and confidence
                                                best_f1_idx = np.argmax(overall_f1_scores)
                                                best_f1 = overall_f1_scores[best_f1_idx]
                                                best_conf = conf_thresholds[best_f1_idx]
                                                
                                                f1_fig.add_trace(go.Scatter(
                                                    x=conf_thresholds,
                                                    y=overall_f1_scores,
                                                    mode='lines',
                                                    name=f"all classes {best_f1:.2f} at {best_conf:.3f}",
                                                    line=dict(width=3, color='black')
                                                ))
                                                
                                                f1_fig.update_layout(
                                                    title='F1-Confidence Curve',
                                                    xaxis_title='Confidence',
                                                    yaxis_title='F1',
                                                    xaxis=dict(range=[0, 1]),
                                                    yaxis=dict(range=[0, 1]),
                                                    height=500,
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(f1_fig, width='stretch')
                                                
                                                # 3. Recall-Confidence Curve
                                                st.markdown("#### Recall-Confidence Curve")
                                                recall_fig = go.Figure()
                                                
                                                for class_name in sorted(all_classes):
                                                    recalls = []
                                                    for conf_thresh in conf_thresholds:
                                                        filtered_preds = [
                                                            p for p in all_predictions
                                                            if p.get('confidence', 0) >= conf_thresh
                                                            and (class_mapping.get(p.get('class', ''), p.get('class', '')) if class_mapping else p.get('class', '')) == class_name
                                                        ]
                                                        tp, fp, fn = match_predictions_to_ground_truth(
                                                            filtered_preds,
                                                            [g for g in ground_truth if g.get('class') == class_name],
                                                            iou_threshold=0.5,
                                                            class_mapping=class_mapping
                                                        )
                                                        tp_count = len([x for x in tp if x.get('class') == class_name])
                                                        fn_count = len([x for x in fn if x.get('class') == class_name])
                                                        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
                                                        recalls.append(recall)
                                                    
                                                    recall_fig.add_trace(go.Scatter(
                                                        x=conf_thresholds,
                                                        y=recalls,
                                                        mode='lines',
                                                        name=class_name,
                                                        line=dict(width=2)
                                                    ))
                                                
                                                # Overall recall curve
                                                overall_recalls = []
                                                for conf_thresh in conf_thresholds:
                                                    filtered_preds = [p for p in all_predictions if p.get('confidence', 0) >= conf_thresh]
                                                    tp, fp, fn = match_predictions_to_ground_truth(
                                                        filtered_preds, ground_truth, iou_threshold=0.5, class_mapping=class_mapping
                                                    )
                                                    total_tp = len(tp)
                                                    total_fn = len(fn)
                                                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                                                    overall_recalls.append(recall)
                                                
                                                recall_fig.add_trace(go.Scatter(
                                                    x=conf_thresholds,
                                                    y=overall_recalls,
                                                    mode='lines',
                                                    name=f"all classes 1.00 at 0.000",
                                                    line=dict(width=3, color='black')
                                                ))
                                                
                                                recall_fig.update_layout(
                                                    title='Recall-Confidence Curve',
                                                    xaxis_title='Confidence',
                                                    yaxis_title='Recall',
                                                    xaxis=dict(range=[0, 1]),
                                                    yaxis=dict(range=[0, 1]),
                                                    height=500,
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(recall_fig, width='stretch')
                                                
                                                # 4. Precision-Confidence Curve
                                                st.markdown("#### Precision-Confidence Curve")
                                                prec_fig = go.Figure()
                                                
                                                for class_name in sorted(all_classes):
                                                    precisions = []
                                                    for conf_thresh in conf_thresholds:
                                                        filtered_preds = [
                                                            p for p in all_predictions
                                                            if p.get('confidence', 0) >= conf_thresh
                                                            and (class_mapping.get(p.get('class', ''), p.get('class', '')) if class_mapping else p.get('class', '')) == class_name
                                                        ]
                                                        tp, fp, fn = match_predictions_to_ground_truth(
                                                            filtered_preds,
                                                            [g for g in ground_truth if g.get('class') == class_name],
                                                            iou_threshold=0.5,
                                                            class_mapping=class_mapping
                                                        )
                                                        tp_count = len([x for x in tp if x.get('class') == class_name])
                                                        fp_count = len([x for x in fp if x.get('class') == class_name])
                                                        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                                                        precisions.append(precision)
                                                    
                                                    prec_fig.add_trace(go.Scatter(
                                                        x=conf_thresholds,
                                                        y=precisions,
                                                        mode='lines',
                                                        name=class_name,
                                                        line=dict(width=2)
                                                    ))
                                                
                                                # Overall precision curve
                                                overall_precisions = []
                                                for conf_thresh in conf_thresholds:
                                                    filtered_preds = [p for p in all_predictions if p.get('confidence', 0) >= conf_thresh]
                                                    tp, fp, fn = match_predictions_to_ground_truth(
                                                        filtered_preds, ground_truth, iou_threshold=0.5, class_mapping=class_mapping
                                                    )
                                                    total_tp = len(tp)
                                                    total_fp = len(fp)
                                                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                                                    overall_precisions.append(precision)
                                                
                                                # Find best precision and confidence
                                                best_prec_idx = np.argmax(overall_precisions)
                                                best_prec = overall_precisions[best_prec_idx]
                                                best_prec_conf = conf_thresholds[best_prec_idx]
                                                
                                                prec_fig.add_trace(go.Scatter(
                                                    x=conf_thresholds,
                                                    y=overall_precisions,
                                                    mode='lines',
                                                    name=f"all classes {best_prec:.2f} at {best_prec_conf:.3f}",
                                                    line=dict(width=3, color='black')
                                                ))
                                                
                                                prec_fig.update_layout(
                                                    title='Precision-Confidence Curve',
                                                    xaxis_title='Confidence',
                                                    yaxis_title='Precision',
                                                    xaxis=dict(range=[0, 1]),
                                                    yaxis=dict(range=[0, 1]),
                                                    height=500,
                                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(prec_fig, width='stretch')
                                                
                                                # 5. Confusion Matrix
                                                st.markdown("#### Confusion Matrix")
                                                
                                                # Build confusion matrix
                                                class_list = sorted(all_classes)
                                                if 'background' not in class_list:
                                                    class_list.append('background')
                                                
                                                cm_size = len(class_list)
                                                confusion_matrix = np.zeros((cm_size, cm_size), dtype=int)
                                                
                                                # Map class names to indices
                                                class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
                                                
                                                # Use matched predictions at IoU 0.5
                                                tp, fp, fn = match_predictions_to_ground_truth(
                                                    all_predictions, ground_truth, iou_threshold=0.5, class_mapping=class_mapping
                                                )
                                                
                                                # Count true positives (correct predictions)
                                                for detection in tp:
                                                    pred_class = detection.get('class', '')
                                                    if pred_class in class_to_idx:
                                                        confusion_matrix[class_to_idx[pred_class], class_to_idx[pred_class]] += 1
                                                
                                                # Count false positives (predicted but not in ground truth)
                                                for detection in fp:
                                                    pred_class = detection.get('class', '')
                                                    if pred_class in class_to_idx:
                                                        # Find the actual class from ground truth (if any)
                                                        gt_class = 'background'
                                                        confusion_matrix[class_to_idx[pred_class], class_to_idx[gt_class]] += 1
                                                
                                                # Count false negatives (in ground truth but not predicted)
                                                for detection in fn:
                                                    gt_class = detection.get('class', '')
                                                    if gt_class in class_to_idx:
                                                        pred_class = 'background'
                                                        confusion_matrix[class_to_idx[pred_class], class_to_idx[gt_class]] += 1
                                                
                                                # Create confusion matrix heatmap
                                                cm_fig = go.Figure(data=go.Heatmap(
                                                    z=confusion_matrix,
                                                    x=class_list,
                                                    y=class_list,
                                                    colorscale='Blues',
                                                    text=confusion_matrix,
                                                    texttemplate='%{text}',
                                                    textfont={"size": 10},
                                                    colorbar=dict(title="Count")
                                                ))
                                                
                                                cm_fig.update_layout(
                                                    title='Confusion Matrix',
                                                    xaxis_title='True',
                                                    yaxis_title='Predicted',
                                                    height=600,
                                                    width=800,
                                                    plot_bgcolor='rgba(255, 255, 255, 0.1)',
                                                    paper_bgcolor='rgba(0, 0, 0, 0)'
                                                )
                                                st.plotly_chart(cm_fig, width='stretch')
                                            
                                            else:
                                                st.info("📊 Install plotly to view interactive charts: `pip install plotly`")
                                        else:
                                            st.warning("No per-class metrics available.")
                                    else:
                                        st.error("No predictions generated. Please check your validation images and model configuration.")

                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                        finally:
                            # Cleanup temporary directory
                            if 'temp_dir' in locals():
                                shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())


