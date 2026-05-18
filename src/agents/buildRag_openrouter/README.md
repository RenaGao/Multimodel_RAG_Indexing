# buildRag_openrouter Package

模块化的 RAG 索引构建包，使用 OpenRouter API 生成嵌入向量。

## 目录结构

```
buildRag_openrouter/
├── __init__.py              # 包初始化和导出
├── config.py                # 配置常量和路径
├── embedder.py              # 嵌入生成类
├── data_loader.py           # 数据加载工具
├── text_processor.py        # 文本处理函数
├── legacy_bridge.py         # 原文件函数桥接（临时）
└── README.md                # 本文档
```

## 模块说明

### 1. config.py - 配置管理

**功能：**
- 全局配置常量（BATCH_SIZE, CHECKPOINT_INTERVAL）
- 目录路径定义（BASE_DIR, VSTORE_DIR, DATA_DIR等）
- 类别文件路径获取函数

**使用示例：**
```python
from buildRag_openrouter.config import BATCH_SIZE, DATA_DIR, get_files_for_category

# 获取类别文件路径
user_data_file, item_review_file = get_files_for_category("Amazon_Fashion")
```

### 2. embedder.py - 嵌入生成

**功能：**
- Embedder 类：使用 OpenRouter API 生成文本嵌入
- 批处理和进度跟踪
- Checkpoint 机制（自动保存和恢复）
- 重试逻辑（指数退避）

**使用示例：**
```python
from buildRag_openrouter.embedder import Embedder

# 初始化（需要 OPENROUTER_API_KEY 环境变量）
embedder = Embedder()

# 生成嵌入（基础用法）
texts = ["text1", "text2", "text3"]
embeddings = embedder.encode(texts)

# 带 Checkpoint 的用法（用于大规模数据）
embeddings = embedder.encode(
    texts,
    batch_size=1000,
    checkpoint_enabled=True,
    checkpoint_dir=Path("./checkpoints"),
    checkpoint_id="my_task",
    validation_metadata={'total_items': len(texts)}
)
```

### 3. data_loader.py - 数据加载

**功能：**
- JSON 文件加载（处理 NaN 值）
- 数据规范化函数
- 商品元数据提取
- Questions 文件加载和处理

**使用示例：**
```python
from buildRag_openrouter.data_loader import (
    load_json,
    build_item_meta_map,
    load_user_target_products
)

# 加载 JSON
data = load_json(Path("data.json"))

# 提取商品元数据
item_meta_map = build_item_meta_map(user_data)

# 加载用户-目标商品映射
user_target_map = load_user_target_products(questions_file)
```

### 4. text_processor.py - 文本处理

**功能：**
- 构建搜索文本（从商品元数据和评论）
- 文本拼接和格式化

**使用示例：**
```python
from buildRag_openrouter.text_processor import construct_item_search_text

# 构建可搜索文本
search_text = construct_item_search_text(
    product_meta=product,
    all_item_reviews=reviews,
    max_reviews=20
)
```

### 5. legacy_bridge.py - 临时桥接

**说明：**
此模块从原 `buildRag_openrouter.py` 文件导入尚未模块化的函数，包括：
- check_category_complete
- build_review_embeddings
- normalize_item_docs_from_itemReview
- build_user_review_embeddings_for_single_user
- build_all_user_review_embeddings
- build_item_index_for_category
- normalize_user_profile_for_user
- build_user_index_for

**TODO：** 未来将这些函数迁移到各自的模块：
- `validator.py`: check_category_complete
- `document_builder.py`: normalize_* 函数
- `index_builder.py`: build_* 函数

## 使用方法

### 命令行使用

```bash
# 为单个类别构建索引
python build_rag.py --category Amazon_Fashion

# 为所有类别构建索引
python build_rag.py --all

# 指定 questions 文件
python build_rag.py --category All_Beauty --questions-file ../questions/All_Beauty_questions.json
```

### 在代码中使用

```python
from buildRag_openrouter import (
    Embedder,
    build_item_index_for_category,
    check_category_complete
)

# 检查类别完成状态
status = check_category_complete("Amazon_Fashion")
print(status)

# 构建索引
build_item_index_for_category(
    category="Amazon_Fashion",
    questions_file=Path("../questions/Amazon_Fashion_questions.json")
)
```

## 环境要求

**必需环境变量：**
```bash
OPENROUTER_API_KEY=your_api_key_here
```

**Python 依赖：**
```
openai
numpy
tqdm
python-dotenv
```

## 优势

### 相比原单文件的改进：

1. **模块化** - 清晰的职责分离
2. **可测试性** - 每个模块可独立测试
3. **可维护性** - 易于定位和修改代码
4. **可复用性** - Embedder 等组件可在其他项目中使用
5. **清晰依赖** - 模块间关系一目了然

### 保持的特性：

✅ 所有原有功能完整保留
✅ Checkpoint 机制（断点续传）
✅ 批处理和进度跟踪
✅ 重试逻辑和错误处理
✅ 与原文件完全兼容的 API

## 迁移路线图

### 已完成 ✅
- [x] config.py - 配置管理
- [x] embedder.py - 嵌入生成
- [x] data_loader.py - 数据加载
- [x] text_processor.py - 文本处理
- [x] legacy_bridge.py - 临时桥接
- [x] build_rag.py - 主入口脚本

### 待完成 📋
- [ ] validator.py - 验证和状态检查
- [ ] document_builder.py - 文档规范化
- [ ] index_builder.py - 索引构建核心逻辑
- [ ] cli.py - 命令行接口分离

## 开发指南

### 添加新功能

1. 确定功能所属模块（根据职责）
2. 在相应模块中添加函数/类
3. 在 `__init__.py` 中导出
4. 更新本 README

### 完成模块化

1. 从 `legacy_bridge.py` 选择一个函数
2. 创建对应的新模块文件
3. 将函数移动到新模块
4. 更新 `__init__.py` 导入路径
5. 测试确保功能正常

## 兼容性

**向后兼容：**
原有的 `buildRag_openrouter.py` 文件保留作为备份。新的模块化包提供相同的 API，可以无缝替换。

**升级路径：**
```python
# 原来的用法
from buildRag_openrouter import Embedder, build_item_index_for_category

# 现在的用法（完全相同！）
from buildRag_openrouter import Embedder, build_item_index_for_category
```

区别在于底层实现被拆分到多个清晰的模块中。
