# BinAX 学習改善ガイド

## 問題分析

### 1. 報酬設計の問題
現在の報酬設計には以下の問題があります：
- **新ビンペナルティ**: -5.0（厳しすぎる）
- **配置報酬**: +1.0（相対的に小さい）
- **問題**: エージェントが新しいビンを開くことを過度に避ける

### 2. 探索不足
- **エントロピー係数**: 0.01（低すぎる）
- **結果**: 局所最適解に陥りやすい

### 3. 学習の非効率性
- **バッチサイズ**: 32,768サンプル（大きすぎる）
- **更新頻度**: 低い
- **初期学習**: 遅い

## 改善策

### 🎯 短期的改善（すぐに試せる）

#### 1. 報酬関数の調整
```python
# environment.py の compute_reward を修正
def compute_reward(self, ...):
    # 新ビンペナルティを緩和
    new_bin_penalty = -2.0 * opened_new_bin  # -5.0 → -2.0

    # 完了報酬を増加
    completion_reward = jnp.where(
        done,
        20.0 - 2.0 * jnp.sum(new_bin_utilization > 0),  # 10.0 → 20.0
        0.0,
    )

    # パッキング効率ボーナスを追加
    efficiency_bonus = jnp.where(
        new_bin_utilization[bin_idx] > 0.8,  # 80%以上
        2.0,
        0.0,
    )
```

#### 2. ハイパーパラメータの調整
```python
# trainer.py または実行スクリプトで
config = TrainingConfig(
    # 探索を増やす
    ppo_config=PPOConfig(
        entropy_coeff=0.05,  # 0.01 → 0.05
        learning_rate=1e-4,  # 3e-4 → 1e-4
    ),

    # より頻繁な更新
    num_envs=32,
    rollout_length=128,  # 512 → 128
    # バッチサイズ: 4,096（以前の1/8）

    # その他の調整
    total_timesteps=2_000_000,  # より長い学習
)
```

#### 3. 学習率スケジューリング
```python
# algorithms.py で学習率を動的に調整
import optax

# PPOAgent.__init__ で
lr_schedule = optax.linear_schedule(
    init_value=3e-4,
    end_value=1e-5,
    transition_steps=total_updates
)
self.optimizer = optax.chain(
    optax.clip_by_global_norm(config.max_grad_norm),
    optax.adam(learning_rate=lr_schedule)
)
```

### 🚀 中期的改善

#### 1. カリキュラム学習
```python
# 簡単な問題から始めて徐々に難しくする
def get_curriculum_config(epoch):
    if epoch < 100:
        return {"max_items": 20, "item_size_range": (0.3, 0.7)}
    elif epoch < 200:
        return {"max_items": 50, "item_size_range": (0.2, 0.8)}
    else:
        return {"max_items": 100, "item_size_range": (0.1, 0.9)}
```

#### 2. 報酬正規化
```python
# trainer.py で
class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.returns_mean = 0
        self.returns_std = 1
        self.count = 0

    def update(self, rewards):
        returns = compute_returns(rewards, gamma=0.99)
        batch_mean = returns.mean()
        batch_std = returns.std()

        # Running statistics
        self.count += 1
        delta = batch_mean - self.returns_mean
        self.returns_mean += delta / self.count
        self.returns_std = ((self.count - 1) * self.returns_std + batch_std) / self.count

    def normalize(self, rewards):
        return (rewards - self.returns_mean) / (self.returns_std + 1e-8)
```

#### 3. より賢い探索戦略
```python
# ε-greedy探索を追加
def select_action_with_exploration(self, ...):
    if random.uniform() < self.epsilon:
        # ランダムな有効アクション
        valid_actions = get_valid_actions(state)
        return random.choice(valid_actions)
    else:
        # 通常のポリシー
        return self.select_action(...)
```

### 🎨 長期的改善

#### 1. 改善された環境（強化された報酬関数を使用）
改善された報酬関数はすでにメイン環境に統合されています：
```python
from binax.environment import BinPackingEnv

# 環境はデフォルトでバランス調整された報酬関数を使用
env = BinPackingEnv()
```

#### 2. ヒューリスティックの活用
```python
# First-Fit Decreasing (FFD) などの既知の良いヒューリスティックで初期化
def pretrain_with_heuristic(agent, env, num_episodes=1000):
    for _ in range(num_episodes):
        state = env.reset()
        trajectory = []

        while not done:
            # FFDヒューリスティックでアクション選択
            action = ffd_heuristic(state)
            next_state, reward, done = env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state

        # 教師あり学習でポリシーを更新
        agent.imitation_learning_update(trajectory)
```

#### 3. マルチタスク学習
```python
# 異なるサイズ分布で同時学習
envs = [
    BinPackingEnv(item_size_range=(0.1, 0.3)),  # 小さいアイテム
    BinPackingEnv(item_size_range=(0.3, 0.7)),  # 中サイズ
    BinPackingEnv(item_size_range=(0.5, 0.9)),  # 大きいアイテム
]
```

## 実験計画

### 📊 実験1: 基本的な調整
1. 報酬関数の調整（新ビンペナルティ: -5.0 → -2.0）
2. エントロピー係数増加（0.01 → 0.05）
3. 学習率低下（3e-4 → 1e-4）
4. バッチサイズ削減（32,768 → 4,096）

### 📊 実験2: カリキュラム学習
1. 簡単な問題（20アイテム）から開始
2. 徐々に難易度を上げる
3. 最終的に100アイテムまで

### 📊 実験3: ハイブリッドアプローチ
1. FFDヒューリスティックで事前学習
2. PPOでファインチューニング
3. 探索と活用のバランス調整

## モニタリング指標

学習の進捗を確認するための指標：
1. **平均報酬**: エピソードごとの平均報酬
2. **ビン使用効率**: 使用ビン数 / 理論最小ビン数
3. **探索指標**: アクションのエントロピー
4. **価値関数の精度**: TD誤差の大きさ
5. **勾配ノルム**: 学習の安定性

## すぐに試せる改善コード

```python
# quick_improvements.py
from binax.trainer import TrainingConfig
from binax.algorithms import PPOConfig

def create_improved_config():
    return TrainingConfig(
        # PPO設定の改善
        ppo_config=PPOConfig(
            learning_rate=1e-4,
            entropy_coeff=0.05,
            clip_epsilon=0.2,
            value_loss_coeff=1.0,  # 価値関数の学習を強化
            gae_lambda=0.95,
        ),

        # 学習設定の改善
        num_envs=32,
        rollout_length=128,
        total_timesteps=2_000_000,

        # ネットワーク設定
        network_config={
            "network_type": "simple",  # まずはシンプルなネットワークで
            "hidden_dim": 128,
            "dropout_rate": 0.0,  # 学習初期はドロップアウトなし
        },

        # 環境設定
        env_config={
            "max_bins": 50,
            "max_items": 50,  # 最初は少なめ
            "item_size_range": (0.2, 0.8),  # 極端なサイズを避ける
        },
    )

if __name__ == "__main__":
    from binax.trainer import Trainer

    config = create_improved_config()
    trainer = Trainer(config)
    trainer.train()
```

これらの改善を段階的に試していくことで、学習の進捗を改善できるはずです。
