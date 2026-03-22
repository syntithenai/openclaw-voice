(function () {
  const FM_API = '/api/file-manager';
  const FM_DEBOUNCE_MS = 500;
  const JSON_EDITOR_MODULE_URL = 'https://cdn.jsdelivr.net/npm/vanilla-jsoneditor/standalone.js';

  function fmEsc(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmMain() {
    return document.getElementById('main');
  }

  function fmState() {
    if (!S.fileManager) {
      S.fileManager = {
        initialized: false,
        loading: false,
        error: '',
        treeByPath: {},
        expandedByPath: { '/': true },
        selectedFolderPath: '/',
        selectedFilePath: '',
        folderChildren: [],
        currentFile: null,
        saveTimersByPath: {},
        saveStateByPath: {},
        activePlainEditorPath: '',
        activeMarkdownEditorPath: '',
        activeJsonEditorPath: '',
        markdownEditor: null,
        jsonEditor: null,
        createFolderModalOpen: false,
        createFolderName: '',
        deleteModalOpen: false,
        deleteTargetType: '',
        deleteTargetPath: '',
        deleteTargetName: '',
        deleteBusy: false,
      };
    }
    return S.fileManager;
  }

  function isVirtualPath(path) {
    const p = String(path || '');
    return p === '/__virtual__/openclaw-config' || p.startsWith('/__virtual__/openclaw-config/');
  }

  function parentPath(path) {
    const p = String(path || '/');
    if (p === '/' || !p.startsWith('/')) return '/';
    const parts = p.split('/').filter(Boolean);
    if (!parts.length) return '/';
    parts.pop();
    return parts.length ? ('/' + parts.join('/')) : '/';
  }

  function updateSaveBadgeDom() {
    const st = fmState();
    const badge = document.getElementById('fmSaveBadge');
    if (!badge) return;
    const file = st.currentFile;
    if (!file) {
      badge.classList.add('hidden');
      badge.textContent = '';
      return;
    }
    const value = String(st.saveStateByPath[file.path] || '');
    if (!value) {
      badge.classList.add('hidden');
      badge.textContent = '';
      return;
    }
    const label = value === 'dirty' ? 'Unsaved'
      : value === 'saving' ? 'Saving'
      : value === 'saved' ? 'Saved'
      : value === 'conflict' ? 'Conflict'
      : 'Error';
    badge.classList.remove('hidden');
    badge.textContent = label;
  }

  async function fmFetchJson(url, opts) {
    const response = await fetch(url, Object.assign({
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
    }, opts || {}));

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = String(payload && payload.error ? payload.error : 'request failed');
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  async function loadTree(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/tree?path=' + encodeURIComponent(path));
    st.treeByPath[path] = Array.isArray(data.children) ? data.children : [];
    return st.treeByPath[path];
  }

  async function loadFolder(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/folder?path=' + encodeURIComponent(path));
    st.folderChildren = Array.isArray(data.children) ? data.children : [];
    st.selectedFolderPath = path;
    return st.folderChildren;
  }

  async function loadFile(path) {
    const st = fmState();
    const data = await fmFetchJson(FM_API + '/file?path=' + encodeURIComponent(path));
    st.currentFile = data;
    st.selectedFilePath = path;
    st.activePlainEditorPath = '';
    st.activeMarkdownEditorPath = '';
    st.activeJsonEditorPath = '';
    return data;
  }

  async function saveFile(path, content) {
    const st = fmState();
    if (!st.currentFile || st.currentFile.path !== path) {
      return;
    }

    st.currentFile.content = String(content || '');
    st.saveStateByPath[path] = 'saving';
    updateSaveBadgeDom();

    try {
      const res = await fmFetchJson(FM_API + '/file?path=' + encodeURIComponent(path), {
        method: 'PUT',
        body: JSON.stringify({
          content: String(content || ''),
          expectedEtag: String(st.currentFile.etag || ''),
        }),
      });
      st.currentFile.etag = res.etag;
      st.currentFile.size = res.size;
      st.currentFile.mtime = res.mtime;
      st.currentFile.content = String(content || '');
      st.saveStateByPath[path] = 'saved';
    } catch (err) {
      st.saveStateByPath[path] = String(err && err.status === 409 ? 'conflict' : 'error');
      st.error = String(err && err.message ? err.message : err);
      renderFileManagerPage(fmMain());
      return;
    }

    updateSaveBadgeDom();
  }

  function queueSave(path, content) {
    const st = fmState();
    if (!path) return;
    if (st.saveTimersByPath[path]) {
      clearTimeout(st.saveTimersByPath[path]);
    }
    if (st.currentFile && st.currentFile.path === path) {
      st.currentFile.content = String(content || '');
    }
    st.saveStateByPath[path] = 'dirty';
    st.saveTimersByPath[path] = setTimeout(() => {
      delete st.saveTimersByPath[path];
      void saveFile(path, content);
    }, FM_DEBOUNCE_MS);
    updateSaveBadgeDom();
  }

  function renderTreeRows(path, depth) {
    const st = fmState();
    const nodes = st.treeByPath[path] || [];
    return nodes.map((node) => {
      const nodePath = String(node.path || '');
      const isFolder = String(node.kind || '') === 'folder' || String(node.kind || '') === 'virtual-folder';
      const isExpanded = !!st.expandedByPath[nodePath];
      const isActive = isFolder ? (st.selectedFolderPath === nodePath) : (st.selectedFilePath === nodePath);
      const rowAction = isFolder ? 'fm-select-folder' : 'fm-select-file';
      const left = 10 + (depth * 14);

      const branch = isFolder
        ? '<button type="button" class="text-xs text-gray-300" data-action="fm-toggle-folder" data-path="' + fmEsc(nodePath) + '">' + (isExpanded ? '-' : '+') + '</button>'
        : '<span class="text-xs text-gray-500">.</span>';

      const row = ''
        + '<div class="fm-tree-row ' + (isActive ? 'active' : '') + '" style="margin-left:' + left + 'px" data-action="' + rowAction + '" data-path="' + fmEsc(nodePath) + '">'
        + branch
        + '<span class="text-sm">' + fmEsc(node.name) + '</span>'
        + '</div>';

      if (isFolder && isExpanded) {
        return row + renderTreeRows(nodePath, depth + 1);
      }
      return row;
    }).join('');
  }

  function renderFolderRows() {
    const st = fmState();
    const items = Array.isArray(st.folderChildren) ? st.folderChildren : [];
    if (!items.length) {
      return '<div class="px-3 py-3 text-sm text-gray-400">Folder is empty</div>';
    }
    return items.map((item) => {
      const p = String(item.path || '');
      const kind = String(item.kind || 'file');
      const active = st.selectedFilePath === p;
      const icon = kind === 'folder' || kind === 'virtual-folder' ? 'DIR' : 'FILE';
      const action = kind === 'folder' || kind === 'virtual-folder' ? 'fm-select-folder' : 'fm-select-file';
      return ''
        + '<div class="fm-file-row ' + (active ? 'active' : '') + '" data-action="' + action + '" data-path="' + fmEsc(p) + '">'
        + '<div class="text-xs text-gray-500">' + icon + '</div>'
        + '<div class="text-sm truncate">' + fmEsc(item.name) + '</div>'
        + '</div>';
    }).join('');
  }

  function currentFolderName() {
    const st = fmState();
    const path = String(st.selectedFolderPath || '/');
    if (path === '/') return 'Workspace';
    const parts = path.split('/').filter(Boolean);
    return parts.length ? parts[parts.length - 1] : 'Workspace';
  }

  function renderEditorPane() {
    const st = fmState();
    const file = st.currentFile;
    if (!file) {
      return '<div class="px-4 py-4 text-sm text-gray-400">Select a file to open an editor/preview.</div>';
    }

    const category = String(file.category || 'binary');
    const editable = !!file.editable;
    const readOnly = editable ? '' : ('<div class="text-xs text-amber-300">' + fmEsc(file.readOnlyReason || 'Read-only file') + '</div>');

    if (category === 'media') {
      const mime = String(file.mimeType || '');
      const src = String(file.previewUrl || '');
      if (mime.startsWith('image/')) {
        return '<div class="fm-preview p-3"><img src="' + fmEsc(src) + '" alt="preview" /></div>';
      }
      if (mime.startsWith('audio/')) {
        return '<div class="fm-preview p-3"><audio controls src="' + fmEsc(src) + '" style="width:100%"></audio></div>';
      }
      if (mime.startsWith('video/')) {
        return '<div class="fm-preview p-3"><video controls src="' + fmEsc(src) + '" style="width:100%"></video></div>';
      }
      return '<div class="fm-preview p-3"><a class="underline" href="' + fmEsc(src) + '" target="_blank" rel="noreferrer noopener">Open media preview</a></div>';
    }

    if (category === 'binary') {
      return ''
        + '<div class="p-4 space-y-2">'
        + '<div class="text-sm text-gray-300">Binary file preview only.</div>'
        + '<a class="underline" href="' + fmEsc(file.previewUrl || '') + '" target="_blank" rel="noreferrer noopener">Open file</a>'
        + '</div>';
    }

    if (category === 'markdown') {
      return ''
        + '<div class="p-3 space-y-2">'
        + readOnly
        + '<textarea id="fmMarkdownEditor" class="fm-textarea">' + fmEsc(file.content || '') + '</textarea>'
        + '</div>';
    }

    if (category === 'json') {
      return ''
        + '<div class="p-3 space-y-2">'
        + readOnly
        + '<div id="fmJsonEditor" class="fm-editor-wrap"></div>'
        + '</div>';
    }

    return ''
      + '<div class="p-3 space-y-2">'
      + readOnly
      + '<textarea id="fmTextEditor" class="fm-textarea">' + fmEsc(file.content || '') + '</textarea>'
      + '</div>';
  }

  function destroyEditors() {
    const st = fmState();
    if (st.markdownEditor && typeof st.markdownEditor.toTextArea === 'function') {
      st.markdownEditor.toTextArea();
      st.markdownEditor = null;
    }
    if (st.jsonEditor && typeof st.jsonEditor.destroy === 'function') {
      st.jsonEditor.destroy();
      st.jsonEditor = null;
    }
    st.activePlainEditorPath = '';
    st.activeMarkdownEditorPath = '';
    st.activeJsonEditorPath = '';
  }

  async function mountEditors() {
    const st = fmState();
    const file = st.currentFile;
    if (!file || S.page !== 'files') return;

    const category = String(file.category || 'binary');
    if (!file.editable) return;

    if (category === 'text') {
      const text = document.getElementById('fmTextEditor');
      if (!text) return;
      if (text.dataset.fmBound !== '1') {
        text.dataset.fmBound = '1';
        text.addEventListener('input', () => {
          queueSave(file.path, text.value);
        });
      }
      st.activePlainEditorPath = file.path;
      return;
    }

    if (category === 'markdown') {
      const textarea = document.getElementById('fmMarkdownEditor');
      if (!textarea || st.activeMarkdownEditorPath === file.path) return;
      if (typeof EasyMDE !== 'function') {
        st.error = 'EasyMDE is not loaded';
        renderFileManagerPage(fmMain());
        return;
      }
      st.activeMarkdownEditorPath = file.path;
      st.markdownEditor = new EasyMDE({
        element: textarea,
        autofocus: false,
        spellChecker: false,
        forceSync: true,
        status: false,
      });
      st.markdownEditor.value(String(file.content || ''));
      st.markdownEditor.codemirror.on('change', () => {
        const next = st.markdownEditor.value();
        queueSave(file.path, next);
      });
      return;
    }

    if (category === 'json') {
      const mount = document.getElementById('fmJsonEditor');
      if (!mount || st.activeJsonEditorPath === file.path) return;

      st.activeJsonEditorPath = file.path;
      let createJSONEditor = null;
      try {
        const mod = await import(JSON_EDITOR_MODULE_URL);
        createJSONEditor = mod.createJSONEditor;
      } catch (err) {
        st.error = 'Failed to load vanilla-jsoneditor: ' + String(err && err.message ? err.message : err);
        renderFileManagerPage(fmMain());
        return;
      }

      let parsed = {};
      try {
        parsed = JSON.parse(String(file.content || '{}'));
      } catch {
        parsed = {};
      }

      st.jsonEditor = createJSONEditor({
        target: mount,
        props: {
          content: { json: parsed },
          mode: 'tree',
          onChange: (updatedContent) => {
            let nextText = '{}';
            if (updatedContent && updatedContent.text !== undefined) {
              nextText = String(updatedContent.text || '{}');
            } else if (updatedContent && updatedContent.json !== undefined) {
              nextText = JSON.stringify(updatedContent.json, null, 2);
            }
            queueSave(file.path, nextText);
          },
        },
      });
    }
  }

  async function ensureInitialized(forceReload) {
    const st = fmState();
    if (!forceReload && st.initialized) return;
    st.loading = true;
    st.error = '';
    renderFileManagerPage(fmMain());
    try {
      await loadTree('/');
      await loadFolder(st.selectedFolderPath || '/');
      st.initialized = true;
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    } finally {
      st.loading = false;
      renderFileManagerPage(fmMain());
      void mountEditors();
    }
  }

  async function selectFolder(path) {
    const st = fmState();
    st.error = '';
    st.selectedFolderPath = path;
    st.selectedFilePath = '';
    st.currentFile = null;
    destroyEditors();
    renderFileManagerPage(fmMain());
    try {
      await loadFolder(path);
      if (!st.treeByPath[path]) {
        await loadTree(path);
      }
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    }
    renderFileManagerPage(fmMain());
  }

  async function selectFile(path) {
    const st = fmState();
    st.error = '';
    st.selectedFilePath = path;
    destroyEditors();
    renderFileManagerPage(fmMain());
    try {
      await loadFile(path);
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
    }
    renderFileManagerPage(fmMain());
    void mountEditors();
  }

  async function toggleTreeFolder(path) {
    const st = fmState();
    st.expandedByPath[path] = !st.expandedByPath[path];
    if (st.expandedByPath[path] && !st.treeByPath[path]) {
      try {
        await loadTree(path);
      } catch (err) {
        st.error = String(err && err.message ? err.message : err);
      }
    }
    renderFileManagerPage(fmMain());
  }

  async function createFolder() {
    const st = fmState();
    const name = String(st.createFolderName || '').trim();
    if (!name) {
      st.error = 'Folder name is required';
      renderFileManagerPage(fmMain());
      return;
    }

    try {
      await fmFetchJson(FM_API + '/folder?path=' + encodeURIComponent(st.selectedFolderPath || '/'), {
        method: 'POST',
        body: JSON.stringify({ name }),
      });
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      delete st.treeByPath[st.selectedFolderPath || '/'];
      await selectFolder(st.selectedFolderPath || '/');
    } catch (err) {
      st.error = String(err && err.message ? err.message : err);
      renderFileManagerPage(fmMain());
    }
  }

  function openDeleteModal(type, path, name) {
    const st = fmState();
    st.deleteModalOpen = true;
    st.deleteTargetType = String(type || '');
    st.deleteTargetPath = String(path || '');
    st.deleteTargetName = String(name || '');
    st.deleteBusy = false;
    renderFileManagerPage(fmMain());
  }

  function closeDeleteModal() {
    const st = fmState();
    st.deleteModalOpen = false;
    st.deleteTargetType = '';
    st.deleteTargetPath = '';
    st.deleteTargetName = '';
    st.deleteBusy = false;
    renderFileManagerPage(fmMain());
  }

  async function confirmDelete() {
    const st = fmState();
    if (st.deleteBusy) return;

    const targetType = String(st.deleteTargetType || '');
    const targetPath = String(st.deleteTargetPath || '');
    if (!targetType || !targetPath) {
      closeDeleteModal();
      return;
    }

    st.deleteBusy = true;
    st.error = '';
    renderFileManagerPage(fmMain());

    try {
      if (targetType === 'file') {
        await fmFetchJson(FM_API + '/file?path=' + encodeURIComponent(targetPath), { method: 'DELETE' });
        if (st.saveTimersByPath[targetPath]) {
          clearTimeout(st.saveTimersByPath[targetPath]);
          delete st.saveTimersByPath[targetPath];
        }
        delete st.saveStateByPath[targetPath];
        st.selectedFilePath = '';
        st.currentFile = null;
        destroyEditors();
        st.treeByPath = {};
        await loadTree('/');
        await selectFolder(parentPath(targetPath));
      } else if (targetType === 'folder') {
        await fmFetchJson(FM_API + '/folder?path=' + encodeURIComponent(targetPath), { method: 'DELETE' });
        st.treeByPath = {};
        await loadTree('/');
        await selectFolder(parentPath(targetPath));
      }
      st.deleteModalOpen = false;
      st.deleteTargetType = '';
      st.deleteTargetPath = '';
      st.deleteTargetName = '';
      st.deleteBusy = false;
      renderFileManagerPage(fmMain());
    } catch (err) {
      st.deleteBusy = false;
      st.deleteModalOpen = false;
      st.error = String(err && err.message ? err.message : err);
      renderFileManagerPage(fmMain());
    }
  }

  function saveBadgeHtml() {
    return '<span id="fmSaveBadge" class="hidden px-2 py-1 rounded text-xs bg-gray-700"></span>';
  }

  window.renderFileManagerPage = function renderFileManagerPage(main) {
    const st = fmState();
    if (!main) return;
    main.dataset.page = 'files';

    const treeRows = renderTreeRows('/', 0);
    const showingFile = !!st.currentFile;
    const folderItems = Array.isArray(st.folderChildren) ? st.folderChildren : [];
    const folderIsEmpty = !showingFile && folderItems.length === 0;
    const selectedFolderPath = String(st.selectedFolderPath || '/');
    const canDeleteEmptyFolder = folderIsEmpty && selectedFolderPath !== '/' && !isVirtualPath(selectedFolderPath);
    const mainPanelBody = showingFile ? renderEditorPane() : renderFolderRows();
    const mainPanelTitle = showingFile
      ? fmEsc(st.currentFile ? st.currentFile.name : 'Editor / Preview')
      : fmEsc(currentFolderName());
    const mainPanelActions = showingFile
      ? '<div class="flex items-center gap-2"><button type="button" class="px-2 py-1 text-xs rounded bg-red-800 hover:bg-red-700" data-action="fm-open-delete-file">Delete</button><button type="button" class="px-2 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600" data-action="fm-close-file">Back to folder</button>' + saveBadgeHtml() + '</div>'
      : ('<div class="flex items-center gap-2">'
        + (canDeleteEmptyFolder ? '<button type="button" class="px-2 py-1 text-xs rounded bg-red-800 hover:bg-red-700" data-action="fm-open-delete-folder">Delete Folder</button>' : '')
        + '<button type="button" class="px-2 py-1 text-xs rounded bg-blue-700 hover:bg-blue-600" data-action="fm-open-create-folder">Create Folder</button>'
        + '</div>');

    let modal = '';
    if (st.deleteModalOpen) {
      const typeLabel = st.deleteTargetType === 'folder' ? 'folder' : 'file';
      const confirmLabel = st.deleteBusy ? 'Deleting...' : 'Delete';
      const disabledAttr = st.deleteBusy ? ' disabled' : '';
      modal = ''
        + '<div class="fm-modal-backdrop">'
        + '<div class="fm-modal space-y-3">'
        + '<div class="text-sm font-semibold">Delete ' + typeLabel + '?</div>'
        + '<div class="text-sm text-gray-300">This permanently deletes <span class="font-semibold">' + fmEsc(st.deleteTargetName || st.deleteTargetPath || typeLabel) + '</span>.</div>'
        + '<div class="flex justify-end gap-2">'
        + '<button type="button" class="px-3 py-1.5 rounded bg-gray-700" data-action="fm-delete-cancel"' + disabledAttr + '>Cancel</button>'
        + '<button id="fmDeleteConfirm" type="button" class="px-3 py-1.5 rounded bg-red-800 hover:bg-red-700" data-action="fm-delete-confirm"' + disabledAttr + '>' + confirmLabel + '</button>'
        + '</div>'
        + '</div>'
        + '</div>';
    } else if (st.createFolderModalOpen) {
      modal = ''
        + '<div class="fm-modal-backdrop">'
        + '<div class="fm-modal space-y-3">'
        + '<div class="text-sm font-semibold">Create folder</div>'
        + '<input id="fmCreateFolderName" type="text" value="' + fmEsc(st.createFolderName || '') + '" class="w-full rounded bg-gray-800 border border-gray-700 px-3 py-2 text-sm" placeholder="Folder name" />'
        + '<div class="flex justify-end gap-2">'
        + '<button type="button" class="px-3 py-1.5 rounded bg-gray-700" data-action="fm-create-cancel">Cancel</button>'
        + '<button type="button" class="px-3 py-1.5 rounded bg-blue-700" data-action="fm-create-confirm">Create</button>'
        + '</div>'
        + '</div>'
        + '</div>';
    }

    if (st.loading) {
      main.innerHTML = '<div class="px-4 py-4 text-sm text-gray-400">Loading file manager...</div>';
      return;
    }

    main.innerHTML = ''
      + '<div class="h-full min-h-0 p-2">'
      + '<div class="fm-layout">'
      + '<section class="fm-panel">'
      + '<div class="px-3 py-2 border-b border-gray-800 text-sm font-semibold">Workspace Tree</div>'
      + '<div class="fm-scroll px-2 py-2">' + treeRows + '</div>'
      + '</section>'
      + '<section class="fm-panel">'
      + '<div class="px-3 py-2 border-b border-gray-800 flex items-center justify-between gap-2">'
      + '<div class="text-sm font-semibold truncate">' + mainPanelTitle + '</div>'
      + mainPanelActions
      + '</div>'
      + '<div class="fm-scroll">' + mainPanelBody + '</div>'
      + '</section>'
      + '</div>'
      + (st.error ? '<div class="px-2 pt-2 text-xs text-red-300">' + fmEsc(st.error) + '</div>' : '')
      + '</div>'
      + modal;

    updateSaveBadgeDom();

    if (st.createFolderModalOpen) {
      setTimeout(() => {
        const input = document.getElementById('fmCreateFolderName');
        if (input) input.focus();
      }, 0);
    }
    if (st.deleteModalOpen) {
      setTimeout(() => {
        const btn = document.getElementById('fmDeleteConfirm');
        if (btn && !st.deleteBusy) btn.focus();
      }, 0);
    }

    setTimeout(() => {
      void mountEditors();
    }, 0);
  };

  window.handleFileManagerClick = function handleFileManagerClick(target, event) {
    const st = fmState();
    const toggle = target.closest('[data-action="fm-toggle-folder"]');
    if (toggle) {
      event.preventDefault();
      event.stopPropagation();
      void toggleTreeFolder(String(toggle.dataset.path || '/'));
      return true;
    }

    const selectFolderBtn = target.closest('[data-action="fm-select-folder"]');
    if (selectFolderBtn) {
      event.preventDefault();
      const path = String(selectFolderBtn.dataset.path || '/');
      void selectFolder(path);
      return true;
    }

    const selectFileBtn = target.closest('[data-action="fm-select-file"]');
    if (selectFileBtn) {
      event.preventDefault();
      const path = String(selectFileBtn.dataset.path || '');
      if (path) void selectFile(path);
      return true;
    }

    const closeFile = target.closest('[data-action="fm-close-file"]');
    if (closeFile) {
      event.preventDefault();
      st.selectedFilePath = '';
      st.currentFile = null;
      destroyEditors();
      renderFileManagerPage(fmMain());
      return true;
    }

    const openCreate = target.closest('[data-action="fm-open-create-folder"]');
    if (openCreate) {
      event.preventDefault();
      st.createFolderModalOpen = true;
      st.createFolderName = '';
      renderFileManagerPage(fmMain());
      return true;
    }

    const openDeleteFile = target.closest('[data-action="fm-open-delete-file"]');
    if (openDeleteFile) {
      event.preventDefault();
      if (st.currentFile && st.currentFile.path) {
        openDeleteModal('file', st.currentFile.path, st.currentFile.name || st.currentFile.path);
      }
      return true;
    }

    const openDeleteFolder = target.closest('[data-action="fm-open-delete-folder"]');
    if (openDeleteFolder) {
      event.preventDefault();
      const path = String(st.selectedFolderPath || '/');
      if (path !== '/' && !isVirtualPath(path)) {
        openDeleteModal('folder', path, currentFolderName());
      }
      return true;
    }

    const cancelCreate = target.closest('[data-action="fm-create-cancel"]');
    if (cancelCreate) {
      event.preventDefault();
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      renderFileManagerPage(fmMain());
      return true;
    }

    const confirmCreate = target.closest('[data-action="fm-create-confirm"]');
    if (confirmCreate) {
      event.preventDefault();
      void createFolder();
      return true;
    }

    const cancelDelete = target.closest('[data-action="fm-delete-cancel"]');
    if (cancelDelete) {
      event.preventDefault();
      if (!st.deleteBusy) closeDeleteModal();
      return true;
    }

    const confirmDeleteBtn = target.closest('[data-action="fm-delete-confirm"]');
    if (confirmDeleteBtn) {
      event.preventDefault();
      if (!st.deleteBusy) {
        void confirmDelete();
      }
      return true;
    }

    return false;
  };

  window.handleFileManagerInput = function handleFileManagerInput(target) {
    const st = fmState();
    if (!target) return false;
    if (target.id === 'fmCreateFolderName') {
      st.createFolderName = String(target.value || '');
      return true;
    }
    return false;
  };

  window.handleFileManagerKeydown = function handleFileManagerKeydown(event) {
    const st = fmState();
    const t = event.target;
    if (!t) return false;
    if (t.id === 'fmCreateFolderName' && event.key === 'Enter') {
      event.preventDefault();
      void createFolder();
      return true;
    }
    if (st.deleteModalOpen && event.key === 'Enter' && !st.deleteBusy) {
      event.preventDefault();
      void confirmDelete();
      return true;
    }
    if (event.key === 'Escape' && st.deleteModalOpen) {
      event.preventDefault();
      if (!st.deleteBusy) closeDeleteModal();
      return true;
    }
    if (event.key === 'Escape' && st.createFolderModalOpen) {
      st.createFolderModalOpen = false;
      st.createFolderName = '';
      renderFileManagerPage(fmMain());
      return true;
    }
    return false;
  };

  window.ensureFileManagerReady = function ensureFileManagerReady() {
    void ensureInitialized(false);
  };

  window.handleFileManagerFsChanged = function handleFileManagerFsChanged(msg) {
    const st = fmState();
    const resync = !!(msg && msg.resyncRequired);
    const changedPaths = Array.isArray(msg && msg.paths) ? msg.paths : [];

    if (resync) {
      st.initialized = false;
      st.treeByPath = {};
      st.folderChildren = [];
      if (S.page === 'files') {
        void ensureInitialized(true);
      }
      return;
    }

    for (const key of Object.keys(st.treeByPath || {})) {
      if (key === '/' || changedPaths.some((p) => String(p || '').startsWith(String(key || '') + '/'))) {
        delete st.treeByPath[key];
      }
    }

    if (S.page === 'files') {
      const folder = String(st.selectedFolderPath || '/');
      void loadTree('/').then(() => loadFolder(folder)).then(() => {
        if (st.selectedFilePath) {
          return loadFile(st.selectedFilePath).catch(() => {
            st.selectedFilePath = '';
            st.currentFile = null;
            return null;
          });
        }
        return null;
      }).finally(() => {
        renderFileManagerPage(fmMain());
      });
    }
  };
})();
