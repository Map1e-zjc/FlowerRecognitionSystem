<script setup lang="ts">
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useUserStore } from '@/stores/user'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const formRef = ref<FormInstance>()
const loading = ref(false)
const form = reactive({ username: '', password: '' })

const rules: FormRules = {
  username: [{ required: true, message: '请输入用户名或邮箱', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

async function submit(): Promise<void> {
  const valid = await formRef.value?.validate().catch(() => false)
  if (!valid) return
  loading.value = true
  try {
    await userStore.login(form.username.trim(), form.password)
    ElMessage.success('登录成功')
    const redirect = (route.query.redirect as string) || '/recognize'
    void router.push(redirect)
  } catch {
    // 错误提示已在拦截器中统一处理
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="page auth-page">
    <el-card class="auth-card">
      <h2 class="page-title">登录</h2>
      <p class="page-subtitle">登录后即可上传花卉照片进行识别，并查看识别历史。</p>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top" @keyup.enter="submit">
        <el-form-item label="用户名或邮箱" prop="username">
          <el-input v-model="form.username" placeholder="请输入用户名或邮箱" clearable />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input v-model="form.password" type="password" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-button type="primary" class="submit" :loading="loading" @click="submit">
          登录
        </el-button>
      </el-form>

      <div class="switch">
        还没有账号？
        <el-button link type="primary" @click="router.push({ name: 'register' })">立即注册</el-button>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.auth-page {
  display: flex;
  justify-content: center;
  padding-top: 56px;
}

.auth-card {
  width: 420px;
}

.submit {
  width: 100%;
}

.switch {
  margin-top: 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
}
</style>
