{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMl94/lA/1tJvgsP2vtU4Wh",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MrBIue/Le-Minh-Thuan-NLP-NLI-TTNTK46-4654100012/blob/main/NLP_NLI_Final.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#1 Cài đặt thư viện\n",
        "!pip install -q transformers datasets evaluate accelerate scikit-learn"
      ],
      "metadata": {
        "id": "LcChIPv4MMMv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "collapsed": true,
        "id": "JryaXkJaMBID"
      },
      "outputs": [],
      "source": [
        "#2 Tải dữ liệu và chia tỷ lệ 80/10/10 chuẩn Zero-shot\n",
        "from datasets import load_dataset\n",
        "\n",
        "print(\"Đang tải dữ liệu...\")\n",
        "# 1. Tải MultiNLI (Tiếng Anh)\n",
        "mnli = load_dataset(\"multi_nli\")\n",
        "# 2. Tải XNLI (Phân vùng Tiếng Việt)\n",
        "xnli = load_dataset(\"xnli\", \"vi\")\n",
        "\n",
        "# --- CHIA TỶ LỆ 80/10/10 ---\n",
        "# MNLI gốc có gần 400k mẫu, Colab miễn phí sẽ không chạy nổi.\n",
        "# Ta lấy một tập con (subset) đại diện theo tỷ lệ 16000 : 2000 : 2000\n",
        "\n",
        "# TRAIN (80%): 16000 mẫu tiếng Anh\n",
        "train_dataset = mnli[\"train\"].shuffle(seed=42).select(range(16000))\n",
        "\n",
        "# VALIDATION (10%): 2000 mẫu tiếng Anh (MNLI dùng tên 'validation_matched')\n",
        "val_dataset = mnli[\"validation_matched\"].shuffle(seed=42).select(range(2000))\n",
        "\n",
        "# TEST (10%): 2000 mẫu tiếng Việt\n",
        "test_dataset = xnli[\"test\"].shuffle(seed=42).select(range(2000))\n",
        "\n",
        "print(f\"Số lượng Train (Tiếng Anh): {len(train_dataset)}\")\n",
        "print(f\"Số lượng Validation (Tiếng Anh): {len(val_dataset)}\")\n",
        "print(f\"Số lượng Test (Tiếng Việt): {len(test_dataset)}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#3 Tiền xử lý với XLM-RoBERTa\n",
        "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
        "\n",
        "model_name = \"xlm-roberta-base\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "\n",
        "# Khởi tạo mô hình phân loại 3 nhãn (Entailment, Neutral, Contradiction)\n",
        "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)\n",
        "\n",
        "def preprocess_function(examples):\n",
        "    # XLM-R sẽ tự động chèn token [SEP] giữa premise và hypothesis\n",
        "    return tokenizer(\n",
        "        examples[\"premise\"],\n",
        "        examples[\"hypothesis\"],\n",
        "        truncation=True,\n",
        "        padding=\"max_length\",\n",
        "        max_length=128\n",
        "    )\n",
        "\n",
        "print(\"Đang mã hóa dữ liệu (Tokenization)...\")\n",
        "# Tokenize tập Anh\n",
        "tokenized_train = train_dataset.map(preprocess_function, batched=True)\n",
        "tokenized_val = val_dataset.map(preprocess_function, batched=True)\n",
        "# Tokenize tập Việt\n",
        "tokenized_test = test_dataset.map(preprocess_function, batched=True)\n",
        "\n",
        "# Lên danh sách các cột thừa có thể có\n",
        "potential_cols_mnli = [\"promptID\", \"pairID\", \"premise\", \"hypothesis\", \"premise_binary_parse\", \"premise_parse\", \"hypothesis_binary_parse\", \"hypothesis_parse\", \"genre\"]\n",
        "potential_cols_xnli = [\"premise\", \"hypothesis\", \"language\"]\n",
        "\n",
        "# TỰ ĐỘNG LỌC: Chỉ xóa những cột thực sự tồn tại trong dataset hiện tại\n",
        "cols_to_remove_mnli = [col for col in potential_cols_mnli if col in tokenized_train.column_names]\n",
        "cols_to_remove_xnli = [col for col in potential_cols_xnli if col in tokenized_test.column_names]\n",
        "\n",
        "# Thực hiện xóa cột thừa\n",
        "tokenized_train = tokenized_train.remove_columns(cols_to_remove_mnli)\n",
        "tokenized_val = tokenized_val.remove_columns(cols_to_remove_mnli)\n",
        "tokenized_test = tokenized_test.remove_columns(cols_to_remove_xnli)\n",
        "\n",
        "# Đổi tên cột 'label' thành 'labels' cho đúng chuẩn đầu vào của PyTorch Trainer\n",
        "tokenized_train = tokenized_train.rename_column(\"label\", \"labels\")\n",
        "tokenized_val = tokenized_val.rename_column(\"label\", \"labels\")\n",
        "tokenized_test = tokenized_test.rename_column(\"label\", \"labels\")\n",
        "\n",
        "print(\"Tiền xử lý hoàn tất! Các cột hiện tại của tập Test:\", tokenized_test.column_names)"
      ],
      "metadata": {
        "id": "f9mDGhgXMf6R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#4 Thiết lập Huấn luyện\n",
        "import numpy as np\n",
        "import evaluate\n",
        "from transformers import TrainingArguments, Trainer\n",
        "\n",
        "# Hàm tính toán độ chính xác\n",
        "metric = evaluate.load(\"accuracy\")\n",
        "def compute_metrics(eval_pred):\n",
        "    logits, labels = eval_pred\n",
        "    predictions = np.argmax(logits, axis=-1)\n",
        "    return metric.compute(predictions=predictions, references=labels)\n",
        "\n",
        "# Cấu hình tham số huấn luyện (Đã cập nhật theo API mới nhất)\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./xlmr_nli_model\",\n",
        "    eval_strategy=\"epoch\",\n",
        "    save_strategy=\"epoch\",\n",
        "    learning_rate=2e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=32,\n",
        "    num_train_epochs=3,\n",
        "    weight_decay=0.01,\n",
        "    load_best_model_at_end=True,\n",
        "    fp16=True,\n",
        "    report_to=\"none\"\n",
        ")\n",
        "\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=tokenized_train,\n",
        "    eval_dataset=tokenized_val,\n",
        "    processing_class=tokenizer,       # ĐÃ SỬA TẠI ĐÂY: Đổi từ tokenizer thành processing_class\n",
        "    compute_metrics=compute_metrics\n",
        ")\n",
        "\n",
        "# BẮT ĐẦU HUẤN LUYỆN\n",
        "print(\"Bắt đầu quá trình Fine-tuning...\")\n",
        "trainer.train()"
      ],
      "metadata": {
        "id": "z4LxctL3Mi0q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#5 Đánh giá Zero-shot trên Tiếng Việt\n",
        "print(\"==================================================\")\n",
        "print(\"BẮT ĐẦU ĐÁNH GIÁ ZERO-SHOT TRÊN TẬP XNLI TIẾNG VIỆT\")\n",
        "print(\"==================================================\")\n",
        "\n",
        "# Yêu cầu mô hình (chỉ mới học tiếng Anh) làm bài thi tiếng Việt\n",
        "test_results = trainer.evaluate(eval_dataset=tokenized_test)\n",
        "\n",
        "print(f\"Độ chính xác (Accuracy) trên tiếng Việt: {test_results['eval_accuracy'] * 100:.2f}%\")"
      ],
      "metadata": {
        "id": "VUCJU8zYMonX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#6 Vẽ Ma trận nhầm lẫn\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "\n",
        "print(\"Đang trích xuất dự đoán chi tiết trên tập Test (Tiếng Việt)...\")\n",
        "# Lấy dự đoán từ mô hình\n",
        "predictions_output = trainer.predict(tokenized_test)\n",
        "y_pred = np.argmax(predictions_output.predictions, axis=1)\n",
        "y_true = predictions_output.label_ids\n",
        "\n",
        "# Tên các nhãn\n",
        "target_names = ['Entailment', 'Neutral', 'Contradiction']\n",
        "\n",
        "# 1. In báo cáo F1-Score, Precision, Recall\n",
        "print(\"\\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---\")\n",
        "print(classification_report(y_true, y_pred, target_names=target_names))\n",
        "\n",
        "# 2. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix)\n",
        "cm = confusion_matrix(y_true, y_pred)\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)\n",
        "plt.xlabel('Dự đoán của Mô hình (Predicted)')\n",
        "plt.ylabel('Nhãn thực tế (True)')\n",
        "plt.title('Ma trận nhầm lẫn trên tập XNLI (Tiếng Việt)')\n",
        "#plt.show()"
      ],
      "metadata": {
        "id": "Wc2sLoXwThtL"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}