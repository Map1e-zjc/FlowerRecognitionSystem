<script setup lang="ts">
import { onMounted } from 'vue'

import AppHeader from '@/components/AppHeader.vue'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

// 刷新页面后用已存 token 拉一次用户信息，失效则自动登出
onMounted(() => {
  if (userStore.isLoggedIn) void userStore.refresh()
})
</script>

<template>
  <div class="app-shell">
    <AppHeader />
    <main>
      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
    </main>
    <footer class="app-footer">
      <span>AI 花卉识别系统 · 基于深度迁移学习（Oxford Flowers-102，102 类）</span>
    </footer>
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  min-height: 100%;
  flex-direction: column;
}

main {
  flex: 1;
}

.app-footer {
  padding: 14px 24px 22px;
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
}
</style>
